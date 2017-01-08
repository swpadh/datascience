package util

import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD


object statistics {

  def print(observations: RDD[Vector]): Unit = {
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println("mean value for each column " + summary.mean)
    println("column-wise variance " + summary.variance)
    println("nonzeros in each column " + summary.numNonzeros)
    println("L1 norm of each column " + summary.normL1)
    println("Euclidean magnitude of each column " + summary.normL2)
    val corPearsonMatrix: Matrix = Statistics.corr(observations, "pearson")
    println("pearson correlation matrix \n" + corPearsonMatrix)
    val corSpearmanMatrix: Matrix = Statistics.corr(observations, "spearman")
    println("spearman correlation matrix \n" + corSpearmanMatrix)
  }

}