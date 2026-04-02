scalaVersion := "2.12.17"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.5.4" % "provided",
  "org.apache.spark" %% "spark-core" % "3.5.4" % "provided",
  "com.softwaremill.sttp.client4" %% "core" % "4.0.0-RC1",
  "com.lihaoyi" %% "upickle" % "4.1.0"
)
