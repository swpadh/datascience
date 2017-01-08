package cluster
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ Encoder, Encoders }
import org.apache.spark.sql.catalyst.encoders.{ RowEncoder, ExpressionEncoder }
import org.apache.spark.sql.expressions.Window._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.mllib.linalg.{ Vectors, Vector, VectorUDT }
import org.apache.spark.mllib.linalg.distributed.{ MatrixEntry, RowMatrix, CoordinateMatrix }
import org.apache.spark.mllib.util._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.clustering.PowerIterationClustering
import org.apache.spark.ml.linalg.{ Vector => mlv }
import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
import org.apache.spark.mllib.clustering.{ BisectingKMeans, BisectingKMeansModel }

object MovieRecommender {

  case class MovieInfo(ID: String, Title: String, ReleaseDate: String, VideoReleaseDate: String, IMDB: String, Unknown: Int, Action: Int, Adventure: Int, Animation: Int, Childrens: Int, Comedy: Int, Crime: Int, Documentary: Int, Drama: Int, Fantasy: Int, FilmNoir: Int, Horror: Int, Musical: Int, Mystery: Int, Romance: Int, SciFi: Int, Thriller: Int, War: Int, Western: Int)

  implicit def bool2int(b: Boolean) = if (b) 1 else 0

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("Recommender")
      .config("spark.sql.warehouse.dir",
        "file:///tmp/spark-warehouse").getOrCreate();

    spark.sparkContext.setLogLevel("ERROR");
    import spark.implicits._

    val base = "src/main/resources/"

    val moviesRDD = spark.sparkContext.textFile(base + "movieLens.txt")

    val movies_df1 = moviesRDD.map { _.split('|').map(_.trim) }.map { cols =>
      var url = java.net.URLDecoder.decode(cols(4), "UTF-8")
      new MovieInfo(cols(0), cols(1), cols(2), cols(3), url, cols(5).toInt, cols(6).toInt, cols(7).toInt, cols(8).toInt, cols(9).toInt, cols(10).toInt, cols(11).toInt, cols(12).toInt, cols(13).toInt, cols(14).toInt, cols(15).toInt, cols(16).toInt, cols(17).toInt, cols(18).toInt, cols(19).toInt, cols(20).toInt, cols(21).toInt, cols(22).toInt, cols(23).toInt)
    }.toDF()

    val movies_df2 = movies_df1.distinct.as[MovieInfo]
    movies_df2.cache()
    println(movies_df2.count)
    println(s"Movies are classified as comedies ${movies_df2.map(_.Comedy).reduce(_ + _)}")

    println(s"Movies are classified as westerner ${movies_df2.map(_.Western).reduce(_ + _)}")

    println(s"Movies are classified as romance and drama ${movies_df2.map(x => bool2int(x.Romance != 0 && x.Drama != 0)).reduce(_ + _)}")

    movies_df2.agg(sum("Action"),
      sum("Adventure"),
      sum("Animation"),
      sum("Childrens"),
      sum("Crime"),
      sum("Comedy"),
      sum("Drama"),
      sum("Fantasy"),
      sum("Horror"),
      sum("Musical"),
      sum("Mystery"),
      sum("Western"),
      sum("Romance"),
      sum("SciFi"),
      sum("Thriller"),
      sum("War")).show

    val andfn = udf((col1: Int, col2: Int) => if (col1 != 0 && col2 != 0) 1 else 0)

    movies_df2.agg(sum(andfn($"Romance", $"Drama"))).show

    val featureCols = movies_df2.columns.slice(5, 24)

    val exprs = featureCols.map(c => col(c).cast("double"))

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val user_ratings = assembler.transform(movies_df2.select(exprs: _*))

    val rowVec: RDD[Vector] = user_ratings.select("features").rdd.map {
      case Row(v: mlv) => Vectors.dense(v.toArray)
    }.cache

    val mat: RowMatrix = new RowMatrix(rowVec)

    val m = mat.numRows()
    val n = mat.numCols()

    println(f"m =$m%d n = $n%d")
    // Compute similar columns perfectly, with brute force.
    val exact = mat.columnSimilarities()

    // Compute similar columns with estimation using DIMSUM
    val approx = mat.columnSimilarities(0.1)

    //approx.entries.map{case MatrixEntry(row: Long, col:Long, value:Double) => Array(row,col,value).mkString(",")}.foreach { x => println(x)}
    //approx.toRowMatrix().rows.map {case v:Vector => v.toArray.mkString(",")}.foreach { x => println(x) }

    val exactEntries = exact.entries.map { case MatrixEntry(i, j, u) => ((i, j), u) }
    val approxEntries = approx.entries.map { case MatrixEntry(i, j, v) => ((i, j), v) }
    val MAE = exactEntries.leftOuterJoin(approxEntries).values.map {
      case (u, Some(v)) =>
        math.abs(u - v)
      case (u, None) =>
        math.abs(u)
    }.mean()

    println(s"Average absolute error in estimate is: $MAE")

    val picmodel = new PowerIterationClustering().setK(2).setMaxIterations(5).run(approx.entries.map(x => (x.i, x.j, x.value)))

    val picclusters = picmodel.assignments.collect().groupBy(_.cluster).mapValues(_.map(_.id))

    val assignments = picclusters.toList.sortBy { case (k, v) => v.length }
    val assignmentsStr = assignments
      .map {
        case (k, v) =>
          s"$k -> ${v.sorted.mkString("[", ",", "]")}"
      }.mkString(", ")
    val sizesStr = assignments.map {
      _._2.length
    }.sorted.mkString("(", ",", ")")
    println(s"Cluster assignments: $assignmentsStr\ncluster sizes: $sizesStr")

    val dist = exact.entries.map { case MatrixEntry(i, j, u) => MatrixEntry(i, j, 2 * Math.acos(1 - u) / Math.PI) }

    val distCoord = new CoordinateMatrix(dist)

    val rowData = distCoord.toRowMatrix().rows.cache

    //distCoord.toRowMatrix().rows.map { case v: Vector => v.toArray.mkString(",") }.foreach { x => println(x) }

    val numClusters = 2
    val numIterations = 20
    val kmeansclusters = KMeans.train(rowData, numClusters, numIterations)

    val WSSSE = kmeansclusters.computeCost(rowData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    val pc: Matrix = distCoord.toRowMatrix().computePrincipalComponents(10)

    val projected: RowMatrix = distCoord.toRowMatrix().multiply(pc)

    val rowDataPc = projected.rows.cache

    val kmeansclustersPc = KMeans.train(rowDataPc, numClusters, numIterations)

    val WSSSEPC = kmeansclustersPc.computeCost(rowDataPc)

    println("Within Set Sum of Squared Errors PC = " + WSSSEPC)

    val bkm = new BisectingKMeans().setK(6)
    val bkmmodel = bkm.run(rowData)

    // Show the compute cost and the cluster centers

    println(s"Compute Cost: ${bkmmodel.computeCost(rowData)}")
    bkmmodel.clusterCenters.zipWithIndex.foreach {
      case (center, idx) =>
        println(s"Cluster Center ${idx}: ${center}")
    }

    spark.sparkContext.stop()
  }

}
