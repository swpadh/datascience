package classification

import scala.collection.mutable.ArrayBuffer

import org.apache.log4j.Level
import org.apache.log4j.LogManager
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext

object TDPrediction {
  import Algorithm._
  def predictLR(labeledData: RDD[LabeledPoint], sc: SparkContext) {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val Array(trainData, testData) = labeledData.randomSplit(Array(0.8, 0.2))
    trainData.cache()
    testData.cache()
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

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
    // util.Plot.jscatterPlot(featureData, 6, "Volume", "Index")
  }
  def predictSVM(labeledData: RDD[LabeledPoint], sc: SparkContext) {
    val Array(trainData, testData) = labeledData.randomSplit(Array(0.8, 0.2))
    trainData.cache()
    testData.cache()

    val numIterations = 100
    val model = SVMWithSGD.train(trainData, numIterations)

    model.clearThreshold()

    val scoreAndLabels = testData.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    val modelPath = "src/main/resources/TDPrediction"
    model.save(sc, modelPath)
    model.toPMML(modelPath + "SVMModel.xml")
    val sameModel = SVMModel.load(sc, modelPath)
  }
  def predictGBT(labeledData: RDD[LabeledPoint], sc: SparkContext) {
    val Array(trainData, testData) = labeledData.randomSplit(Array(0.8, 0.2))
    trainData.cache()
    testData.cache()
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 10
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainData, boostingStrategy)

    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification GBT model:\n" + model.toDebugString)

    val modelPath = "src/main/resources/TDPrediction"
    model.save(sc, modelPath)
    val sameModel = GradientBoostedTreesModel.load(sc,
      modelPath)

  }
  def predictRF(labeledData: RDD[LabeledPoint], sc: SparkContext) {
    val sqlContext = new SQLContext(sc)
    val Array(trainData, testData) = labeledData.randomSplit(Array(0.8, 0.2))
    trainData.cache()
    testData.cache()

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification forest model:\n" + model.toDebugString)
    val modelPath = "src/main/resources/TDPrediction"

    model.save(sc, modelPath)
    val sameModel = RandomForestModel.load(sc, modelPath)

  }
  def predictDT(labeledData: RDD[LabeledPoint], sc: SparkContext) {
    val sqlContext = new SQLContext(sc)
    val Array(trainData, testData) = labeledData.randomSplit(Array(0.8, 0.2))
    trainData.cache()
    testData.cache()
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)

    val modelPath = "src/main/resources/TDPrediction"
    model.save(sc, modelPath)
    val sameModel = DecisionTreeModel.load(sc, modelPath)
  }
  def main(args: Array[String]): Unit = {

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
    val sc = new SparkContext(new SparkConf().setAppName("TDPrediction").setMaster("local[*]"))

    val base = "src/main/resources/"
    val rawBankData = sc.textFile(base + "bank.csv")

    val header = rawBankData.takeOrdered(1)
    header.foreach {
      line =>
        val names = line.split(";")
        names.foreach { x => print(x + " ") }
        println()
    }
    val OHEDict: Map[(Int, String), Int] = createOHEDictionary

    val bankData = rawBankData.subtract(sc.makeRDD(header))

    val labeledData = parseOHEPoint(bankData, OHEDict, 64).distinct
    val algorithm: Algorithm = SVM
    algorithm match {
      case SVM => predictSVM(labeledData, sc)
      case LR  => predictLR(labeledData, sc)
      case GBT => predictGBT(labeledData, sc)
      case RF  => predictRF(labeledData, sc)
      case DT  => predictDT(labeledData, sc)
    }

    sc.stop()
  }
  def parseOHEPoint(bankData: RDD[String], OHEDict: Map[(Int, String), Int], numOHEFeats: Int): RDD[LabeledPoint] = {

    bankData.map { line =>

      val rawFeatures = line.split(';')
      val label = rawFeatures.last.replace("\"", "")
      val fdFeatures = rawFeatures.slice(0, rawFeatures.length - 1)
      var features = scala.collection.mutable.ArrayBuffer[(Int, String)]()
      for (featureId <- 0 to fdFeatures.length - 1) {
        features.append((featureId, fdFeatures(featureId).replace("\"", "")))
      }

      val oneOHEFeat = oneHotEncoding(features.toArray, OHEDict, numOHEFeats)
      label match {
        case "yes" =>
          LabeledPoint(0, oneOHEFeat)
        case "no" =>
          LabeledPoint(1.0, oneOHEFeat)
      }
    }
  }

  def oneHotEncoding(rawFeats: Array[(Int, String)], OHEDict: Map[(Int, String), Int], numOHEFeats: Int): SparseVector = {
    var features = scala.collection.mutable.ArrayBuffer[(Int, Double)]()
    for (x <- rawFeats) {
      x match {
        case (0, _)  => features.append((0, x._2 toDouble)) //Age
        case (5, _)  => features.append((5, x._2 toDouble)) //balance
        case (9, _)  => features.append((37, x._2 toDouble)) //day 
        case (11, _) => features.append((50, x._2 toDouble)) //duration: last contact duration
        case (12, _) => features.append((51, x._2 toDouble)) //campaign: number of contacts performed during this campaign and for this client
        case (13, _) => features.append((52, x._2 toDouble)) //pdays: number of days that passed by after the client was last contacted from a previous campaign
        case (14, _) => features.append((53, x._2 toDouble)) //previous: number of contacts performed before this campaign and for this client
        case (16, _) => features.append((57, x._2 toDouble)) //emp.var.rate: employment variation rate
        case (17, _) => features.append((58, x._2 toDouble)) //cons.price.idx: consumer price index
        case (18, _) => features.append((59, x._2 toDouble)) //cons.conf.idx: consumer confidence index 
        case (19, _) => features.append((60, x._2 toDouble)) //euribor3m: euribor 3 month rate
        case (20, _) => features.append((61, x._2 toDouble)) //nr.employed: number of employees 
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
    dict put ((1, "admin."), 1) //job
    dict put ((1, "blue-collar"), 2)
    dict put ((1, "entrepreneur"), 3)
    dict put ((1, "housemaid"), 4)
    dict put ((1, "management"), 5)
    dict put ((1, "retired"), 6)
    dict put ((1, "self-employed"), 7)
    dict put ((1, "services"), 8)
    dict put ((1, "student"), 9)
    dict put ((1, "technician"), 10)
    dict put ((1, "unemployed"), 11)
    dict put ((1, "unknown"), 12)
    dict put ((2, "divorced"), 13) //marital
    dict put ((2, "married"), 14)
    dict put ((2, "single"), 15)
    dict put ((2, "unknown"), 16)
    dict put ((3, "basic.4y"), 17) //education
    dict put ((3, "basic.6y"), 18)
    dict put ((3, "basic.9y"), 19)
    dict put ((3, "high.school"), 20)
    dict put ((3, "illiterate"), 21)
    dict put ((3, "professional.course"), 22)
    dict put ((3, "university.degree"), 23)
    dict put ((3, "unknown"), 24)
    dict put ((4, "no"), 25) //default
    dict put ((4, "yes"), 26)
    dict put ((4, "unknown"), 27)
    dict put ((6, "no"), 29) //housing
    dict put ((6, "yes"), 30)
    dict put ((6, "unknown"), 31)
    dict put ((7, "no"), 32) //loan
    dict put ((7, "yes"), 33)
    dict put ((7, "unknown"), 34)
    dict put ((8, "cellular"), 35) //contact
    dict put ((8, "telephone"), 36)
    dict put ((10, "jan"), 38) // month: last contact month of year 
    dict put ((10, "feb"), 39)
    dict put ((10, "mar"), 40)
    dict put ((10, "apr"), 41)
    dict put ((10, "may"), 42)
    dict put ((10, "jun"), 43)
    dict put ((10, "jul"), 44)
    dict put ((10, "aug"), 45)
    dict put ((10, "sep"), 46)
    dict put ((10, "oct"), 47)
    dict put ((10, "nov"), 48)
    dict put ((10, "dec"), 49)
    dict put ((15, "failure"), 54) //poutcome: outcome of the previous marketing campaign
    dict put ((15, "nonexistent"), 55)
    dict put ((15, "success"), 56)
    dict.toMap
  }
}