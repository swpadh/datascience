package graph

import org.apache.spark.sql.SparkSession;
import org.apache.spark.graphx._
import org.apache.spark.sql.Row
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.catalyst.encoders.{RowEncoder,ExpressionEncoder}

object Flights {
  case class Flight(dayOfMonth: Int, dayOfWeek: Int, carrier: String, tailNum: String, flightNum: Int, originId: Int, originCode: String, destId: Int, destCode: String, schedDeptime: Int, actualDeptime: Int, depDelaymins: Int, schedArrtime: Int, actArrtime: Int, arrivalDelay: Int, elapsedTime: Int, dist: Int)

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("Flights")
      .config("spark.sql.warehouse.dir",
        "file:///tmp/spark-warehouse").getOrCreate();

    spark.sparkContext.setLogLevel("ERROR");

    import spark.implicits._

    val base = "src/main/resources/"

    val airport_df = spark.read.option("header", "true").option("inferSchema", "true").csv(base + "flights_data.csv")
    airport_df.sample(false, 5E-5, 0).show()
    airport_df.printSchema()

    val flightsRDD = airport_df.map {
      case Row(dayOfMonth: Int, dayOfWeek: Int, carrier: String, tailNum: String, flightNum: Int, originId: Int, originCode: String, destId: Int, destCode: String, schedDeptime: Int, actualDeptime: Int, depDelaymins: Int, schedArrtime: Int, actArrtime: Int, arrivalDelay: Int, elapsedTime: Int, dist: Int) =>
        Flight(dayOfMonth, dayOfWeek, carrier, tailNum, flightNum, originId,

          originCode, destId, destCode, schedDeptime, actualDeptime,
          depDelaymins, schedArrtime, actArrtime, arrivalDelay, elapsedTime, dist)
    }.cache()
    
    val encoder = Encoders.tuple(Encoders.scalaLong, Encoders.STRING)
    
    val airports = flightsRDD.map(flight => (flight.originId.toLong, flight.originCode))(encoder).distinct
        
    val airport = airports.take(1)
    println(airport.apply(0)._1 + "  " + airport.apply(0)._2)

    val encoder2 = Encoders.tuple(Encoders.tuple(Encoders.scalaInt,Encoders.scalaInt),Encoders.scalaInt,Encoders.scalaInt )
    
    val routes = flightsRDD.map(flight => ((flight.originId, flight.destId), flight.dist, flight.arrivalDelay))(encoder2).distinct

    routes.cache
    
    val encoder4 = Encoders.kryo[Edge[(Int,Int)]]

    val edges = routes.map {case ((originId, destId), distance, arrivalDelay) => Edge[(Int,Int)](originId.toLong, destId.toLong, (distance, arrivalDelay))}(encoder4)

    val graph = Graph(airports.rdd, edges.rdd)

    graph.vertices.take(5).foreach(println)
    graph.edges.takeSample(false, 5).foreach(println)

    airport_df.filter(airport_df("originCode") === "BOS").select(airport_df("originCode"), airport_df("originId")).distinct().show()
    val airport_id = graph.vertices.filter { case (originId, originCode) => originCode == "BOS" }.map { case (originId, originCode) => "( " + originCode + "," + originId + " )" }.take(1).foreach(println)

    val nTrips = graph.edges.filter { case (Edge(originId, destId, (distance, arrivalDelay))) => distance > 1000 }.count
    println("No of trips greater than 1000 =" + nTrips)

    val gg = graph.mapEdges(e => 50.toDouble + e.attr._1.toDouble / 20)
    gg.triplets.map { x => (x.srcAttr, x.attr) }.take(1).foreach(println)

    val initialGraph = gg.mapVertices((id, _) => if (id == 10397) 0.0 else Double.PositiveInfinity)
    initialGraph.triplets.map { x => (x.srcAttr, x.attr) }.take(1).foreach(println)

    graph.triplets.filter(x => x.srcAttr == "BOS" && x.attr._1 > 1000).distinct().sortBy(x => x.attr, false).take(5).foreach(println)

    println("\nBusiest airport:")
    val busyAirport = graph.triplets.map { x => (x.srcAttr, x.attr._1) }.reduceByKey(_ + _)
    busyAirport.sortBy(-_._2).take(1).foreach(println)

    println(" The number of flights from Boston To Los Angeles = " + graph.triplets.filter(x => x.srcAttr == "BOS" && x.dstAttr == "LAX").count())
    def max(a: (VertexId, Int), b: (VertexId, Int)): (VertexId, Int) = {
      if (a._2 > b._2) a else b
    }

    println("Max in-degree = " + graph.inDegrees.reduce(max))
    println("Max out-degree = " + graph.outDegrees.reduce(max))
    val maxDegrees = graph.degrees.reduce(max)
    println("Max total degrees = " + maxDegrees)

    println("The airport with the most inbound and outbound flights is " + graph.triplets.filter(x => x.srcId == maxDegrees._1).map(x => x.srcAttr).first())

    
    val encoder3 = Encoders.kryo[(Long, String)]
    
   
    val airportMap = airports.map { case (originId, originCode) => (originId -> originCode) }(encoder3).collect.toMap

    println("Airport that has the most incoming flights ");
    val maxIncoming = graph.inDegrees.collect.sortWith(_._2 > _._2).map(x => (airportMap(x._1), x._2)).take(3)
    maxIncoming.foreach(println)

    println("Airport that has the most outgoing flights ");
    val maxout = graph.outDegrees.join(airports.rdd).sortBy(_._2._1, ascending = false).take(3)
    maxout.foreach(println)

    println("Airport that has the most outgoing flights ");
    val maxOutgoing = graph.outDegrees.collect.sortWith(_._2 > _._2).map(x => (airportMap(x._1), x._2)).take(3)
    maxOutgoing.foreach(println)

    println("Top 10 flights from airport to airport ");
    val flights = graph.triplets.sortBy(_.attr, ascending = false).map(triplet =>
      "There were " + triplet.attr.toString + " flights from " + triplet.srcAttr + " to " + triplet.dstAttr + ".").take(10)
    flights.foreach(println)

    val longestDelay = airport_df.groupBy().max("arrivalDelay")
    longestDelay.show()

    println("On-time / Early Flights: " + graph.edges.filter { case (Edge(originId, destId, (distance, arrivalDelay))) => arrivalDelay <= 0 }.count())
    println("Delayed Flights: " + graph.edges.filter { case (Edge(originId, destId, (distance, arrivalDelay))) => arrivalDelay >= 0 }.count())

    //Calculate PageRanks in descending order sort by ranking
    val ranks = graph.pageRank(tol = 0.1).vertices
    ranks.sortBy(x => x._2, false).take(1).foreach(println)

    //Determine airports by airport code in PageRank descending order
    val ranksByAirport = ranks.join(airports.rdd).distinct().sortBy(x => x._2._1, false)
    println("The most important airports by PageRank are")
    ranksByAirport.map(x => x._2._2).take(5).foreach(println)
    
    spark.sparkContext.stop()
  }
}