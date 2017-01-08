package classification

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions

object Algorithm extends Enumeration {
  type Algorithm = Value
  val SVM, LR,GBT,RF,DT = Value
}

object RegType extends Enumeration {
  type RegType = Value
  val L1, L2 = Value
}

object CreditApproval {

  import Algorithm._
  import RegType._

  case class Params(
    input: String = null,
    numIterations: Int = 100,
    stepSize: Double = 1.0,
    algorithm: Algorithm = LR,
    regType: RegType = L2,
    regParam: Double = 0.01)

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("CreditApproval").setMaster("local[*]"))
    val base = "src/main/resources/"
    val rawApprovalData = sc.textFile(base + "crx.data")

    val OHEDict: Map[(Int, String), Int] = createOHEDictionary

    val data = parseOHEPoint(rawApprovalData, OHEDict, 47).distinct

    val defaultParams = Params()

    run(data)

    sc.stop()
  }
  def run(data: RDD[LabeledPoint]): Unit =
    {
      val Array(training, cv, test) = data.randomSplit(Array(0.8, 0.1, 0.1))

      training.cache()
      cv.cache()
      test.cache()

      var bestModel: LogisticRegressionModel = null
      var bestLogLoss = 1e10

      val stepSizes = List(1, 10)
      val regParams = List(1e-6, 1e-3)

      val algorithm = new LogisticRegressionWithSGD()

      // fixed hyperparameters
      algorithm.optimizer
        .setNumIterations(50)
        .setRegParam(1e-6)
        .setStepSize(10.0)
        .setUpdater(new SquaredL2Updater())
      algorithm.setIntercept(true)

      for (stepSize <- stepSizes) {
        for (regParam <- regParams) {

          algorithm.optimizer.setNumIterations(50).setRegParam(regParam).setStepSize(stepSize)
          var model = algorithm.run(training);
          var logLossVa = evaluateResults(model, cv)
          println(f"stepSize = $stepSize%.1f, regParam = $regParam%.0e logLossVa = $logLossVa%.3f")
          if (logLossVa < bestLogLoss) {
            bestModel = model
            bestLogLoss = logLossVa
          }
        }
      }

      predict(test, bestModel)

      training.unpersist()
      cv.unpersist()
      test.unpersist()
    }
  def predict(data: RDD[LabeledPoint], model: LogisticRegressionModel): Unit =
    {
      val prediction = model.predict(data.map(_.features))
      val predictionAndLabel = prediction.zip(data.map(_.label))

      val metrics = new BinaryClassificationMetrics(predictionAndLabel)

      val precision = metrics.precisionByThreshold
      precision.foreach {
        case (t, p) =>
          println(s"Threshold: $t, Precision: $p")
      }

      val recall = metrics.recallByThreshold
      recall.foreach {
        case (t, r) =>
          println(s"Threshold: $t, Recall: $r")
      }

      val PRC = metrics.pr

      val f1Score = metrics.fMeasureByThreshold
      f1Score.foreach {
        case (t, f) =>
          println(s"Threshold: $t, F-score: $f, Beta = 1")
      }

      val beta = 0.5
      val fScore = metrics.fMeasureByThreshold(beta)
      f1Score.foreach {
        case (t, f) =>
          println(s"Threshold: $t, F-score: $f, Beta = 0.5")
      }
      val auPRC = metrics.areaUnderPR
      println("Area under precision-recall curve = " + auPRC)
      val thresholds = precision.map(_._1)
      val roc = metrics.roc
      val auROC = metrics.areaUnderROC
      println("Area under ROC = " + auROC)
    }
  def parseOHEPoint(approvalData: RDD[String], OHEDict: Map[(Int, String), Int], numOHEFeats: Int): RDD[LabeledPoint] = {

    approvalData.map { line =>
      val rawFeatures = line.split(',')
      val label = rawFeatures.last
      val approvalFeatures = rawFeatures.slice(0, rawFeatures.length - 1)
      var features = scala.collection.mutable.ArrayBuffer[(Int, String)]()
      for (featureId <- 0 to approvalFeatures.length - 1) {
        features.append((featureId, approvalFeatures(featureId)))
      }

      val oneOHEFeat = oneHotEncoding(features.toArray, OHEDict, numOHEFeats)
      label match {
        case "+" =>
          LabeledPoint(0, oneOHEFeat)
        case "-" =>
          LabeledPoint(1.0, oneOHEFeat)
      }
    }
  }

  def oneHotEncoding(rawFeats: Array[(Int, String)], OHEDict: Map[(Int, String), Int], numOHEFeats: Int): SparseVector = {
    var features = scala.collection.mutable.ArrayBuffer[(Int, Double)]()
    for (x <- rawFeats) {
      x match {
        case (1, _)  => features.append((2, x._2 toDouble)) //Age
        case (2, _)  => features.append((3, x._2 toDouble)) //Debt
        case (7, _)  => features.append((34, x._2 toDouble)) //YearsEmployed
        case (10, _) => features.append((39, x._2 toDouble)) //CreditScore
        case (13, _) => features.append((45, x._2 toDouble)) //ZipCode
        case (14, _) => features.append((46, x._2 toDouble)) //Income
        case _ => OHEDict.get(x) match {
          case Some(s) => features.append((s, 1))
          case None    =>
        }
      }
    }
    val fmap = features.toMap
    new SparseVector(numOHEFeats, fmap.keys.toArray, fmap.values.toArray)
  }

  def createOHEDictionary(): Map[(Int, String), Int] = {
    var dict = scala.collection.mutable.Map[(Int, String), Int]()
    dict put ((0, "b"), 0) 
    dict put ((0, "a"), 1)
    dict put ((3, "u"), 4) 
    dict put ((3, "y"), 5)
    dict put ((3, "l"), 6)
    dict put ((3, "t"), 7)
    dict put ((4, "g"), 8) 
    dict put ((4, "p"), 9)
    dict put ((4, "gg"), 10)
    dict put ((5, "c"), 11) 
    dict put ((5, "d"), 12)
    dict put ((5, "cc"), 13)
    dict put ((5, "i"), 14)
    dict put ((5, "j"), 15)
    dict put ((5, "k"), 16)
    dict put ((5, "m"), 17)
    dict put ((5, "r"), 18)
    dict put ((5, "q"), 19)
    dict put ((5, "w"), 20)
    dict put ((5, "x"), 21)
    dict put ((5, "e"), 22)
    dict put ((5, "aa"), 23)
    dict put ((5, "ff"), 24)
    dict put ((6, "v"), 25) 
    dict put ((6, "h"), 26)
    dict put ((6, "bb"), 27)
    dict put ((6, "j"), 28)
    dict put ((6, "n"), 29)
    dict put ((6, "z"), 30)
    dict put ((6, "dd"), 31)
    dict put ((6, "ff"), 32)
    dict put ((6, "o"), 33)
    dict put ((8, "t"), 35) 
    dict put ((8, "f"), 36)
    dict put ((9, "t"), 37) 
    dict put ((9, "f"), 38)
    dict put ((11, "t"), 40) 
    dict put ((11, "f"), 41)
    dict put ((12, "g"), 42) 
    dict put ((12, "p"), 43)
    dict put ((12, "s"), 44)
    dict.toMap
  }
  def computeLogLoss(p: Double, y: Double): Double =
    {
      val epsilon = 10e-12
      val pi = p.toInt
      val yi = y.toInt

      val x = pi match {
        case 0 => p + epsilon
        case 1 => p - epsilon
      }
      yi match {
        case 1 => -math.log(x)
        case 0 => -math.log(1 - x)
      }
    }
  def getP(x: SparseVector, w: DenseVector, intercept: Double): Double =
    {
      import util.BLAS._
      var rawPrediction = intercept + dot(w, x)
      //Bound the raw prediction value
      rawPrediction = math.min(rawPrediction, 20)
      rawPrediction = math.max(rawPrediction, -20)
      1.0 / (1.0 + math.exp(-rawPrediction))
    }

  def evaluateResults(model: GeneralizedLinearModel, data: RDD[LabeledPoint]): Double =
    {
      (data.map(lp => computeLogLoss(getP(lp.features.asInstanceOf[SparseVector], model.weights.asInstanceOf[DenseVector], model.intercept), lp.label))).mean()
    }
  def baselineLogLoss(data: RDD[LabeledPoint]): Double =
    {
      val classOneFrac = (data.map(lp => lp.label)).mean()
      (data.map(lp => computeLogLoss(classOneFrac, lp.label))).mean()
    }
}