package regression

import scala.Ordering
import scala.reflect.runtime.universe

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

object PredictMPG {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("PredictMPG").setMaster("local[*]"))

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val base = "src/main/resources/"
    val rawAutoData = sc.textFile(base + "Auto.csv")

    val header = rawAutoData.takeOrdered(1)(Ordering[String].reverse)
    header.foreach {
      line =>
        val names = line.split(",")
        names.foreach { x => print(x + " ") }
        println()
    }

    val labeledData = rawAutoData.subtract(sc.makeRDD(header)).map { line =>
      val rawFeatures = line.split(',').dropRight(1).map(_.toDouble)
      val label = rawFeatures.head
      val featureVector = Vectors.dense(rawFeatures.tail)
      LabeledPoint(label, featureVector)
    }

    val Array(trainData, cvData, testData) = labeledData.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    val polyEx = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyfeatures")

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setSolver("l-bfgs")
      .setFeaturesCol("polyfeatures")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(polyEx.degree, Array(2, 5))
      .addGrid(lr.fitIntercept)
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(polyEx, lr))

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    println("CrossValidator parameters:\n" + cv.explainParams() + "\n")

    val cvModel = cv.fit(trainData.toDF())

    println("cvModel Estimator Parameter : " + cvModel.estimatorParamMaps)
    val paramMap = cvModel.extractParamMap()
    paramMap.toSeq.foreach {
      case ParamPair(_, v) =>
        if (v.isInstanceOf[Pipeline]) {
          val t = v.asInstanceOf[Pipeline]
          t.extractParamMap().toSeq.foreach {
            case ParamPair(_, v) => println(" Pipeline = " + v)
          }
          println(t.getStages.apply(1).asInstanceOf[LinearRegression].explainParams())
        } else if (v.isInstanceOf[RegressionEvaluator]) {
          val t = v.asInstanceOf[RegressionEvaluator]
          t.extractParamMap().toSeq.foreach {
            case ParamPair(_, v) => println(" RegressionEvaluator = " + v)
          }
        }
    }

    val predictions = cvModel.transform(testData.toDF())
    predictions.show()
    predictions.select("features", "label", "prediction")
      .collect()
      .foreach {
        case Row(features: Vector, label: Double, prediction: Double) =>
          println(s"($features, $label) -> prediction=$prediction")
      }

    sc.stop()
  }
}