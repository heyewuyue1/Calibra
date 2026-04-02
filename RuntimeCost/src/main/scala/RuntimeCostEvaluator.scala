package cn.edu.ruc

import org.apache.log4j.LogManager
import org.apache.spark.SparkConf
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{Cost, CostEvaluator, SimpleCost}

case class RuntimeCostEvaluator(conf: SparkConf) extends CostEvaluator {
  private val logger =  LogManager.getLogger(this.getClass)
  override def evaluateCost(plan: SparkPlan): Cost = {
    logger.warn(s"Evaluating cost for:\n${plan.toJSON.take(100)}")
    // Actually plan is a PhysicalPlan
    val req = CostRequest(
      `type` = 2,
      candidates = List(
        PlanInfo(
          plan = plan.toString(),
          queryStages = Map.empty,
          card = -1L,
          size = -1L
        )
      ),
      advisoryChoose = 0
    )
    val costs = WebUtils.sendCostRequest(req)
    SimpleCost(costs.get.head.toLong)
  }
}


