package graph

import org.apache.tinkerpop.gremlin.structure.io.graphml._
import org.apache.tinkerpop.gremlin.structure._
import org.apache.tinkerpop.gremlin.structure.io._
import org.apache.tinkerpop.gremlin.tinkergraph.structure._
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.{ Graph => sg }
import scala.collection.mutable.ListBuffer

object GraphMLReader {

  def main(args: Array[String]): Unit = {
    val trGraph: Graph = TinkerGraph.open()
    trGraph.io(IoCore.graphml()).readGraph("src/main/resources/fraud")
    val verIter = trGraph.vertices()
    val edgeIter = trGraph.edges()
    val sc = new SparkContext(new SparkConf().setAppName("CreditApproval").setMaster("local[*]"))

    var verList = new ListBuffer[Vertex]()

    while (verIter.hasNext()) {
      verList += verIter.next()
    }

    val verRDD = sc.parallelize(verList)

    var edgeList = new ListBuffer[Edge]()

    while (edgeIter.hasNext()) {
      edgeList += edgeIter.next()
    }

    val edgeRDD = sc.parallelize(edgeList)

  }
}