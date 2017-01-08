package classification

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

object Metrics {

  def print(predictions: DataFrame, sc: SparkContext): Unit =
    {
      def predictionAndLabel(): RDD[(Double, Double)] =
        {
          val dataPrediction: ArrayBuffer[Double] = ArrayBuffer[Double]()
          val dataLabel: ArrayBuffer[Double] = ArrayBuffer[Double]()

          predictions.select("prediction", "label").collect().foreach {
            case Row(prediction: Double, label: Double) =>
              dataPrediction += prediction
              dataLabel += label
          }
          sc.makeRDD(dataPrediction.zip(dataLabel))
        }

      val metrics = new BinaryClassificationMetrics(predictionAndLabel)

      println(s"Test areaUnderPR = ${metrics.areaUnderPR()}")
      println(s"Test areaUnderROC = ${metrics.areaUnderROC()}")
      println(s"numBins = ${metrics.numBins}.")
      val thresholds = metrics.thresholds().collect().mkString(" ")
      println(s"thresholds = ${thresholds}.")

      metrics.roc().foreach {
        case (tp, fp) =>
          println(s"True Positive: $tp, False Positive: $fp")
      }

      metrics.pr().foreach {
        case (p, r) =>
          println(s"Precision: $p, Recall: $r")
      }

      val beta = 0.5
      metrics.fMeasureByThreshold(beta).foreach {
        case (t, f) =>
          println(s"Threshold: $t, F-score: $f, Beta = 0.5")
      }

      metrics.fMeasureByThreshold().foreach {
        case (t, f) =>
          println(s"Threshold: $t, F-score: $f, Beta = 1")
      }
      metrics.precisionByThreshold().foreach {
        case (t, p) =>
          println(s"Threshold: $t, Precision: $p")
      }
      metrics.recallByThreshold().foreach {
        case (t, r) =>
          println(s"Threshold: $t, Recall: $r")
      }
    }
  def print(summary: BinaryLogisticRegressionTrainingSummary): Unit =
    {
      println(s"Test areaUnderROC = ${summary.areaUnderROC}")

      summary.fMeasureByThreshold.foreach {
        r =>
          r match {
            case Row(threshold: Double, f: Double) =>
              println(s"Threshold: $threshold, F-score: $f")
          }
      }
      summary.pr.foreach { r =>
        r match {
          case Row(recall: Double, precision: Double) =>
            println(s"Precision: $precision, Recall: $recall")
        }
      }
      summary.precisionByThreshold.foreach { r =>
        r match {
          case Row(threshold: Double, precision: Double) =>
            println(s"Threshold: $threshold, Precision: $precision")
        }
      }
      summary.recallByThreshold.foreach { r =>
        r match {
          case Row(threshold: Double, recall: Double) =>
            println(s"Threshold: $threshold, Recall: $recall")
        }
      }
      summary.roc.foreach { r =>
        r match {
          case Row(fpr: Double, tpr: Double) =>
            println(s"FPR: $fpr, TPR: $tpr")
        }
      }
    }

}