package cn.edu.ruc

import org.apache.log4j.LogManager
import org.apache.spark.SparkConf
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{Cost, CostEvaluator, SimpleCost}

case class EqualCostEvaluator(conf: SparkConf) extends CostEvaluator {
  private val logger =  LogManager.getLogger(this.getClass)
  override def evaluateCost(plan: SparkPlan): Cost = {
    SimpleCost(0L)
  }
}
