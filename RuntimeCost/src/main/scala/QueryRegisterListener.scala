package cn.edu.ruc

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.util.QueryExecutionListener

class QueryRegisterListener extends QueryExecutionListener with Logging {
  override def onSuccess(funcName: String, qe: QueryExecution, durationNs: Long): Unit = {
    logWarning(s"durationNs: $durationNs")
    val sessionName = qe.sparkSession.sparkContext.getConf.get("spark.app.name")
    WebUtils.sendRegisterRequest(sessionName, qe.executedPlan.toString, durationNs)
  }

  override def onFailure(funcName: String, qe: QueryExecution, exception: Exception): Unit = {
    // 处理查询执行失败的情况
    logError(s"Query execution failed: ${exception.getMessage}")
    val sessionName = qe.sparkSession.sparkContext.getConf.get("spark.app.name")
    WebUtils.sendRegisterRequest(sessionName, qe.executedPlan.toString, -1)
  }
}