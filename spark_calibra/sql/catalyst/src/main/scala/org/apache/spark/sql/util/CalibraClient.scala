/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.util

import org.apache.logging.log4j.{LogManager, Logger}
import org.apache.spark.sql.catalyst.optimizer.JoinReorderDP.JoinPlan
import org.apache.spark.sql.internal.SQLConf
import upickle.default._
import sttp.client4.quick._
import sttp.model.HttpVersion.HTTP_1_1
import sttp.model.StatusCode._

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


object CalibraClient {
  val logger: Logger = LogManager.getLogger(this.getClass)
  val conf: SQLConf = SQLConf.get
  val serverUri: String = conf.getConfString("spark.sql.calibra.serverUri",
    "http://10.77.110.152:10533")

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

  def betterThan(thisPlan: JoinPlan, otherPlan: JoinPlan, advisoryChoose: Boolean): Boolean = {
    val req = CostRequest(
      `type` = 0,
      candidates = List(
        PlanInfo(
          plan = thisPlan.plan.toString,
          queryStages = Map.empty,
          card = thisPlan.planCost.card.toLong,
          size = thisPlan.planCost.size.toLong
        ),
        PlanInfo(
          plan = otherPlan.plan.toString,
          queryStages = Map.empty,
          card = otherPlan.planCost.card.toLong,
          size = otherPlan.planCost.size.toLong
        )
      ),
      advisoryChoose = if (advisoryChoose) 0 else 1
    )

    val costsOpt = sendCostRequest(req)
    val costs = costsOpt.get
    costs(0) < costs(1)
  }

  def chooseBestPlan(planType: Int, plans: List[Any], advisoryChoose: Int = 0): List[Double] = {
    val planInfos: List[PlanInfo] =
      plans.map {
        case jp: JoinPlan => PlanInfo(
          plan = jp.toString,
          queryStages = Map.empty,
          card = jp.planCost.card.toLong,
          size = jp.planCost.size.toLong
        )
        case other => PlanInfo(
          plan = other.toString,
          queryStages = Map.empty,
          card = -1L,
          size = -1L
        )
      }
    val req = CostRequest(
      `type` = planType,
      candidates = planInfos,
      advisoryChoose = advisoryChoose
    )
    val costsOpt = sendCostRequest(req)
    val costs = costsOpt.getOrElse(List.fill(plans.length)(1.0)
      .updated(advisoryChoose, 0.0))
    costs
  }
}

