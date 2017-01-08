package classification

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

object MarketPrediction {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("MarketPrediction").setMaster("local[*]"))

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val base = "src/main/resources/"
    val rawSmarketData = sc.textFile(base + "Smarket.csv")

    val header = rawSmarketData.takeOrdered(1)
    header.foreach {
      line =>
        val names = line.split(",")
        names.foreach { x => print(x + " ") }
        println()
    }

    val labeledData = rawSmarketData.subtract(sc.makeRDD(header)).map { line =>
      val items = line.split(',')
      val rawFeatures = items.drop(1).dropRight(1).map(_.toDouble)
      val tick = items.last.trim()
      val label =
        if (tick.indexOf("Up") >= 0)
          1.0
        else
          0.0
      val featureVector = Vectors.dense(rawFeatures.init)
      LabeledPoint(label, featureVector)
    }

    val featureData = labeledData.map { x => x.features }

    val Array(trainData, cvData, testData) = labeledData.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
    //println(trainData.toDF().schema)
    val lrModel = lr.fit(trainData.toDF())

    println("lrModel was fit using parameters: " + lrModel.explainParams)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    println(s"featuresCol: ${trainingSummary.featuresCol}")
    println(s"labelCol: ${trainingSummary.labelCol}")
    println(s"probabilityCol: ${trainingSummary.probabilityCol}")

    val predictions = lrModel.transform(testData.toDF())

    predictions.show()
    predictions.select("features", "label", "prediction", "probability")
      .collect()
      .foreach {
        case Row(features: Vector, label: Double, prediction: Double, probability: Vector) =>
          println(s"($features, $label) -> (prediction=$prediction, probability=$probability)")
      }

    //util.statistics.print(featureData)
    Metrics.print(predictions, sc)
    Metrics.print(trainingSummary.asInstanceOf[BinaryLogisticRegressionTrainingSummary])
    val dataX: ArrayBuffer[Double] = ArrayBuffer[Double]()
    predictions.select("features")
      .collect().foreach {
        case Row(features: Vector) =>
          dataX += features.apply(6)
      }
    //util.Plot.scatterPlot(featureData,6, "Volume", "Index")
    util.Plot.jscatterPlot(featureData, 6, "Volume", "Index")
    sc.stop()
  }

}