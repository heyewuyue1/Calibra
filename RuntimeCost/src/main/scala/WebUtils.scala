package cn.edu.ruc

import sttp.client4.quick._
import sttp.model.HttpVersion.HTTP_1_1
import sttp.model.StatusCode.Ok
import upickle.default._
import org.apache.log4j.LogManager

case class QueryStageInfo(
                           materialized: Boolean,
                           card: Long,
                           size: Long,
                           stagePlan: String
                         )
object QueryStageInfo { implicit val rw: ReadWriter[QueryStageInfo] = macroRW }

case class PlanInfo(
                     plan: String,
                     queryStages: Map[String, QueryStageInfo],
                     card: Long,
                     size: Long
                   )
object PlanInfo { implicit val rw: ReadWriter[PlanInfo] = macroRW }

case class CostRequest(
                        `type`: Int,
                        candidates: List[PlanInfo],
                        advisoryChoose: Int
                      )
object CostRequest { implicit val rw: ReadWriter[CostRequest] = macroRW }

case class CostResponse(costs: List[Double])
object CostResponse { implicit val rw: ReadWriter[CostResponse] = macroRW }

case class RegisterReqBody(
  sessionName: String, finalPlan: String, executionTime: Long
)

object WebUtils {
  private val logger =  LogManager.getLogger(this.getClass)
  private val serverUri = System.getProperty("spark.sql.adaptive.costEvaluator.serverUri", "http://10.77.110.152:10533")

  def sendCostRequest(req: CostRequest): Option[List[Double]] = {

    val bodyStr = write(req)
    try {
      val response = quickRequest
        .post(uri"$serverUri/cost")
        .header("Content-Type", "application/json")
        .body(bodyStr)
        .httpVersion(HTTP_1_1)
        .send()
      if (response.code == Ok) {
        val parsed = read[CostResponse](response.body)
        Some(parsed.costs)
      } else None

    } catch { case e: Exception =>
      e.printStackTrace()
      None
    }
  }

  def sendRegisterRequest(sessionName: String, finalPlan: String, executionTime: Long): Unit = {
    val reqBody = RegisterReqBody(sessionName, finalPlan, executionTime)
    implicit val ownerReg: ReadWriter[RegisterReqBody] = macroRW
    val registerBodyStr: String = write(reqBody)
    try {
      quickRequest
        .post(uri"$serverUri/register")
        .header("Content-Type", "application/json")
        .httpVersion(HTTP_1_1)
        .body(registerBodyStr)
        .send()
      logger.warn("Query execution registered")
    } catch {
      case e: Exception => logger.error("Error register query execution", e)
    }
  }
}
