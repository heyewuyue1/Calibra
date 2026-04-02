package cn.edu.ruc

import org.apache.spark.sql.catalyst.plans.logical.{Join, JoinHint, LogicalPlan, Project, UnaryNode}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeSet, ExpressionSet, PredicateHelper}
import org.apache.spark.sql.catalyst.optimizer.Cost
import org.apache.spark.sql.catalyst.optimizer.JoinReorderDP.JoinPlan
import org.apache.spark.sql.execution.adaptive.{LogicalQueryStage, QueryStageExec}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf
import scala.math._

object OptimizationCost {
  var CPlan = 0L
}

class SubquerySelection extends Rule[LogicalPlan] with Logging with PredicateHelper {

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val start = System.currentTimeMillis()
    val finalPlan = reorderTopLevelJoin(plan)
    OptimizationCost.CPlan += System.currentTimeMillis() - start
    finalPlan
  }

  // 对最顶层的 Join 节点调用 reorderJoin
  private def reorderTopLevelJoin(plan: LogicalPlan): LogicalPlan = {
    // 递归找到最顶层的 Join 节点，并替换为 reorderJoin 的结果
    def rewrite(plan: LogicalPlan): LogicalPlan = plan match {
      case join: Join =>
        reorderJoinLeading(join)
      case u: UnaryNode =>
        u.withNewChildren(Seq(rewrite(u.child)))
      case other =>
        other // 非 UnaryNode 且不是 Join，直接返回
    }
    rewrite(plan)
  }

  private def reorderJoinLeading(plan: LogicalPlan): LogicalPlan = {
    val (innerJoins, conditions) = extractInnerJoins(plan)
    logWarning(s"Inner joins: ${innerJoins.mkString(", ")}".replace("\n", ""))
    logWarning(s"Conditions: ${conditions.mkString(", ")}".replace("\n", ""))
    if (innerJoins.length < 2) {
      logWarning("No need to reorder join")
      return plan
    }
    // 在这里用betterThan决定出一个leading的inner Join
    val req = innerJoins.toList.map {
      case lqs @ LogicalQueryStage(_, stage: QueryStageExec) =>
        PlanInfo(
          plan = lqs.toString(),
          queryStages = Map(
            lqs.toString().trim -> QueryStageInfo(
              materialized = stage.isMaterialized,
              card = stage.getRuntimeStatistics.rowCount.map(_.toLong).getOrElse(-1L),
              size = stage.getRuntimeStatistics.sizeInBytes.toLong,
              stagePlan = stage.plan.toString()
            )
          ),
          card = lqs.stats.rowCount.map(_.toLong).getOrElse(-1L),
          size = lqs.stats.sizeInBytes.toLong
        )

      case p =>
        PlanInfo(
          plan = p.toString(),
          queryStages = Map.empty,
          card = p.stats.rowCount.map(_.toLong).getOrElse(-1L),
          size = p.stats.sizeInBytes.toLong
        )
    }
    val costListOpt = WebUtils.sendCostRequest(CostRequest(0, req, 0))
    // 返回cost最小的那个index
    if (costListOpt.isEmpty) {
      logWarning("Cost list is None, fallback to original plan")
      return plan
    }
    logWarning(s"Cost list: $costListOpt")
    val costList = costListOpt.get
    val leading = costList.zipWithIndex.minBy(_._1)._2
    logWarning(s"Choose leading index: $leading, cost: ${costList(leading)}")

    val front = innerJoins(leading)
    val rest = innerJoins.zipWithIndex.filterNot { case (_, idx) => idx == leading }.map(_._1)
    var reordered = Seq(front) ++ rest
    logWarning(s"Initial reordered: $reordered")  // 输出: List(d, b, a, c, e)

    var currentPlan = JoinPlan(Set(), reordered.head, ExpressionSet(), Cost(0, 0))
    reordered = reordered.patch(0, Nil, 1)

    while (reordered.nonEmpty) {
      var i = 0
      var joined = false
      while (i < reordered.length && !joined) {
        val candidate = reordered(i)
        val candidatePlan = JoinPlan(Set(), candidate, ExpressionSet(), Cost(0, 0))
        buildJoin(currentPlan, candidatePlan, conditions, plan.outputSet) match {
          case Some(p) =>
            currentPlan = p
            reordered = reordered.patch(i, Nil, 1)
            joined = true
            logWarning(s"Build new join plan: $currentPlan")
            logWarning(s"Reordered: $reordered")
          case None =>
            i += 1
        }
      }
      if (!joined) {
        logWarning("Cannot find joinable plan from reordered (which should not happen), return original plan")
        return plan
      }
    }
    currentPlan.plan
  }

  private def extractInnerJoins(plan: LogicalPlan): (Seq[LogicalPlan], ExpressionSet) = {
    plan match {
      case Join(left, right, _, Some(cond), _) =>
        val (leftPlans, leftConditions) = extractInnerJoins(left)
        val (rightPlans, rightConditions) = extractInnerJoins(right)
        (leftPlans ++ rightPlans, leftConditions ++ rightConditions ++
          splitConjunctivePredicates(cond))
      case Project(projectList, j @ Join(_, _, _, _, _))
        if projectList.forall(_.isInstanceOf[Attribute]) =>
        extractInnerJoins(j)
      case _ =>
        (Seq(plan), ExpressionSet())
    }
  }

  /**
   * Builds a new JoinPlan if the following conditions hold:
   * - the sets of items contained in left and right sides do not overlap.
   * - there exists at least one join condition involving references from both sides.
   *
   * @param oneJoinPlan One side JoinPlan for building a new JoinPlan.
   * @param otherJoinPlan The other side JoinPlan for building a new join node.
   * @param conf SQLConf for statistics computation.
   * @param conditions The overall set of join conditions.
   * @param topOutput The output attributes of the final plan.
   * @param filters Join graph info to be used as filters by the search algorithm.
   * @return Builds and returns a new JoinPlan if both conditions hold. Otherwise, returns None.
   */
  private def buildJoin(
                         oneJoinPlan: JoinPlan,
                         otherJoinPlan: JoinPlan,
                         conditions: ExpressionSet,
                         topOutput: AttributeSet): Option[JoinPlan] = {

    if (oneJoinPlan.itemIds.intersect(otherJoinPlan.itemIds).nonEmpty) {
      // Should not join two overlapping item sets.
      return None
    }

    val onePlan = oneJoinPlan.plan
    val otherPlan = otherJoinPlan.plan
    val joinConds = conditions
      .filterNot(l => canEvaluate(l, onePlan))
      .filterNot(r => canEvaluate(r, otherPlan))
      .filter(e => e.references.subsetOf(onePlan.outputSet ++ otherPlan.outputSet))
    if (joinConds.isEmpty) {
      // Cartesian product is very expensive, so we exclude them from candidate plans.
      // This also significantly reduces the search space.
      logWarning(s"Cartesian product detected between $onePlan and $otherPlan, skipping this join.")
      return None
    }

    // Put the deeper side on the left, tend to build a left-deep tree.
    val (left, right) = if (oneJoinPlan.itemIds.size >= otherJoinPlan.itemIds.size) {
      (onePlan, otherPlan)
    } else {
      (otherPlan, onePlan)
    }
    val newJoin = Join(left, right, Inner, joinConds.reduceOption(And), JoinHint.NONE)
    val collectedJoinConds = joinConds ++ oneJoinPlan.joinConds ++ otherJoinPlan.joinConds
    logWarning(s"collectedJoinConds: $collectedJoinConds")
    val remainingConds = conditions -- collectedJoinConds
    logWarning(s"remainingConds: $remainingConds")
    val neededAttr = AttributeSet(remainingConds.flatMap(_.references)) ++ topOutput
    val neededFromNewJoin = newJoin.output.filter(neededAttr.contains)
    logWarning(s"neededFromNewJoin: $neededFromNewJoin")
    val newPlan =
      if ((newJoin.outputSet -- neededFromNewJoin).nonEmpty) {
        Project(neededFromNewJoin, newJoin)
      } else {
        newJoin
      }

    val itemIds = oneJoinPlan.itemIds.union(otherJoinPlan.itemIds)
    // Now the root node of onePlan/otherPlan becomes an intermediate join (if it's a non-leaf
    // item), so the cost of the new join should also include its own cost.
    val newPlanCost = oneJoinPlan.planCost + otherJoinPlan.planCost
    Some(JoinPlan(itemIds, newPlan, collectedJoinConds, newPlanCost))
  }


  private def betterThan(thisCost: Cost, otherCost: Cost, conf: SQLConf): Boolean = {
    if (otherCost.card == 0 || otherCost.size == 0) {
      false
    } else {
      val relativeRows = BigDecimal(thisCost.card) / BigDecimal(otherCost.card)
      val relativeSize = BigDecimal(thisCost.size) / BigDecimal(otherCost.size)
      Math.pow(relativeRows.doubleValue, conf.joinReorderCardWeight) *
        Math.pow(relativeSize.doubleValue, 1 - conf.joinReorderCardWeight) < 1
    }
  }
}
