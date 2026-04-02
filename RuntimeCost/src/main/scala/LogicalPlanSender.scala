package cn.edu.ruc

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.logging.log4j.LogManager
import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.execution.adaptive.{LogicalQueryStage, QueryStageExec}


class LogicalPlanSender(session: SparkSession, runTime: Boolean) extends Rule[LogicalPlan] {
    val logger = LogManager.getLogger(this.getClass)
    override def apply(plan: LogicalPlan): LogicalPlan = {
        if (!runTime) {
          logger.warn(s"Sending logical plan:\n${plan.toString()}")
          val req = CostRequest(
            `type` = 0,
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
          WebUtils.sendCostRequest(req)
        } else {
          if (runTime) {
            val executedStages = plan.collect {
              case lqs@LogicalQueryStage(_, stage: QueryStageExec) =>
                logWarning(s"Stage String: ${stage.plan.toString()}")
                lqs.toString.trim -> (
                  if (stage.isMaterialized) true else false,
                  stage.getRuntimeStatistics.rowCount, // rowCount
                  stage.getRuntimeStatistics.sizeInBytes, // sizeInBytes
                  stage.plan.toString() // plan
                )
            }.toMap
            logger.warn(s"Sending Partially Executed Plan:\n${plan.toString()}")
            val req = CostRequest(
              `type` = 2,
              candidates = List(
                PlanInfo(
                  plan = plan.toString(),
                  queryStages = executedStages.map { case (k, v) =>
                    k -> QueryStageInfo(
                      materialized = v._1,
                      card = v._2.get.toLong,
                      size = v._3.toLong,
                      stagePlan = v._4
                    )
                  },
                  card = -1L,
                  size = -1L
                )
              ),
              advisoryChoose = 0
            )
            WebUtils.sendCostRequest(req)
          }
        }
        plan
    }
}

class LogicalPlanSenderInjector extends (SparkSessionExtensions => Unit) {
  private val logger = LogManager.getLogger(this.getClass)
  override def apply(extensions: SparkSessionExtensions): Unit = {
        extensions.injectRuntimeOptimizerRule(session => {
          new LogicalPlanSender(session, true)
        })
        extensions.injectOptimizerRule(session => {
          new LogicalPlanSender(session, false)
        })
        logger.warn("LogicalPlanSender is injected")
        val SSAEnabled = System.getProperty("spark.sql.adaptive.lssa.enabled", "false")
        if (SSAEnabled == "true") {
          extensions.injectRuntimeOptimizerRule(_ => {
            new SubquerySelection()
          })
          logger.warn("Learned SSA is injected")
        }
    }
}
