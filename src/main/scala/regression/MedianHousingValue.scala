package regression

import scala.Ordering
import scala.reflect.runtime.universe

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

object MedianHousingValue {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("MedianHousingValue").setMaster("local[*]"))

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val base = "src/main/resources/"
    val rawHousingData = sc.textFile(base + "Boston.csv")

    val header = rawHousingData.takeOrdered(1)(Ordering[String].reverse)
    header.foreach {
      line =>
        val names = line.split(",")
        names.foreach { x => print(x + " ") }
        println()
    }

    val labeledData = rawHousingData.subtract(sc.makeRDD(header)).map { line =>
      val rawFeatures = line.split(',').drop(1).map(_.toDouble)
      val label = rawFeatures.last
      val featureVector = Vectors.dense(rawFeatures.init)
      LabeledPoint(label, featureVector)
    }

    val Array(trainData, cvData, testData) = labeledData.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setSolver("l-bfgs")

    println("LinearRegression parameters:\n" + lr.explainParams() + "\n")

    val lrModel = lr.fit(trainData.toDF())

    println("lrModel was fit using parameters: " + lrModel.explainParams)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2 = 1 - (RSS/TSS): ${trainingSummary.r2}")
    //println(s"pvalues: ${trainingSummary.pValues}")
    //println(s"tvalues: ${trainingSummary.tValues}")

    val predictions = lrModel.transform(testData.toDF())
    predictions.show()
    predictions.select("features", "label", "prediction")
      .collect()
      .foreach {
        case Row(features: Vector, label: Double, prediction: Double) =>
          println(s"($features, $label) -> prediction=$prediction")
      }
    util.Plot.regressionPlot(labeledData, 12,"lstat","medv", lrModel)
    sc.stop()
  }
}