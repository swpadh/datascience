package regression

import scala.Long
import scala.reflect.runtime.universe

import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.sum
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType

import breeze.stats.distributions.StudentsT

class Metrics(predictionAndObservations: RDD[(Double, Double)], lModel: LinearRegressionModel, predictions: DataFrame) {

  private[Metrics] def this(predictionAndObservations: DataFrame, lModel: LinearRegressionModel, predictions: DataFrame) =
    this(predictionAndObservations.rdd.map(r => (r.getDouble(0), r.getDouble(1))), lModel, predictions)

  private lazy val summary: MultivariateStatisticalSummary = {
    val summary: MultivariateStatisticalSummary = predictionAndObservations.map {
      case (prediction, observation) => Vectors.dense(observation, observation - prediction)
    }.aggregate(new MultivariateOnlineSummarizer())(
      (summary, v) => summary.add(v),
      (sum1, sum2) => sum1.merge(sum2))
    summary
  }

  private lazy val SSy = math.pow(summary.normL2(0), 2)
  //
  //Residual Sum of Squares(RSS/SSR) or sum of squared errors of prediction (SSE) 
  //
  private lazy val RSS = math.pow(summary.normL2(1), 2)
  //
  //Total Sum of Squares (TSS/SST)
  //
  private lazy val TSS = summary.variance(0) * (summary.count - 1)
  //
  //Standard Error of Mean or model sum of squares, or the explained sum of squares
  //
  private lazy val SE = {
    val yMean = summary.mean(0)
    predictionAndObservations.map {
      case (prediction, _) => math.pow(prediction - yMean, 2)
    }.sum()
  }

  private lazy val numInstances: Long = predictions.count()

  private lazy val  metrics: RegressionMetrics = {
    new RegressionMetrics(
      predictions
        .select(col("prediction"), col("label").cast(DoubleType))
        .rdd
        .map { case Row(pred: Double, label: Double) => (pred, label) })
  }

 private lazy val degreesOfFreedom: Long = if (lModel.getFitIntercept) {
    numInstances - lModel.coefficients.size - 1
  } else {
    numInstances - lModel.coefficients.size
  }
   private lazy val  coefficientStandardErrors: Array[Double] = {
    val rss =
      if (!lModel.isDefined(lModel.weightCol) || lModel.getWeightCol.isEmpty) {
        metrics.meanSquaredError * numInstances
      } else {
        val t = udf { (pred: Double, label: Double, weight: Double) =>
          math.pow(label - pred, 2.0) * weight
        }
        predictions.select(t(col(lModel.getPredictionCol), col(lModel.getLabelCol),
          col(lModel.getWeightCol)).as("wse")).agg(sum(col("wse"))).first().getDouble(0)
      }
    val sigma2 = rss / degreesOfFreedom
    summary.mean.toArray //TODO
  }

  private lazy val  tStatistics: Array[Double] = {
    val estimate = if (lModel.getFitIntercept) {
      Array.concat(lModel.coefficients.toArray, Array(lModel.intercept))
    } else {
      lModel.coefficients.toArray
    }
    estimate.zip(coefficientStandardErrors).map { x => x._1 / x._2 }
  }
  private lazy val  pValues: Array[Double] = {
    tStatistics.map { x => 2.0 * (1.0 - StudentsT(degreesOfFreedom.toDouble).cdf(math.abs(x))) }
  }
 
 
  def print(predictions: DataFrame): Unit = {

    println("variance explained by regression " + metrics.explainedVariance)
    println("mean absolute error = " + metrics.meanAbsoluteError)
    println("mean squared error = " + metrics.meanSquaredError)
    println("root mean squared error = " + metrics.rootMeanSquaredError)
    println("r2 = " + metrics.r2)
  }


}