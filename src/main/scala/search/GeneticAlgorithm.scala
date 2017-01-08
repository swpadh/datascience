package search

object GeneticAlgorithm {

  type allele = Boolean
  type chromosome = Array[allele]
  type population = Array[individual]

  var ncross = 0 //number of crossovers
  var nmutation = 0 //number of mutations

  case class individual(var chrom: chromosome, val x: Double, var fitness: Double, var parent1: Int, var parent2: Int, var xsite: Int)
  case class bestever(var chrom: chromosome, var fitness: Double, var generation: Int)

  def main(args: Array[String]): Unit = {
    val popsize = 100 //Population size
    val lchrom = 20 //length of the chromosome per individual
    val maxgen = 100 //Maximum generation
    val pcross = 0.5.asInstanceOf[Double] //Crossover probability
    val pmutation = 0.1.asInstanceOf[Double] //Mutation probability

    var oldpop = new Array[individual](popsize) //last generation of individuals
    var newpop = new Array[individual](popsize) //next generation of individuals
    var bestfit = new bestever(new Array[allele](lchrom), 0.0.asInstanceOf[Double], 0) //fittest individual so far

    for (j <- 0 until popsize) {
      oldpop(j) = new individual(new Array[allele](lchrom), 0.0.asInstanceOf[Double], 0.0.asInstanceOf[Double], 0, 0, 0)
      newpop(j) = new individual(new Array[allele](lchrom), 0.0.asInstanceOf[Double], 0.0.asInstanceOf[Double], 0, 0, 0)
    }
    ncross = 0
    nmutation = 0
    initPopulation(oldpop, objfunc)
    //printPopulation(oldpop)

    for (gen <- 0 until maxgen) {
      println(f" GENERATION $gen%d-> $maxgen%d");

      generation(oldpop, newpop, pcross, pmutation, objfunc)

      var stat = statistics(oldpop, bestfit, gen)

      printStatistics(stat._5, gen, stat._1, stat._2, stat._3, stat._4)

      var temp = oldpop;
      oldpop = newpop;
      newpop = temp;
    }
  }

  def printPopulation(pop: population): Unit =
    {
      pop.foreach { apop =>
        apop.chrom.foreach { alel =>
          print(if (alel == false) 0 else 1)
        }
        println
        var fitness = apop.fitness
        var x = apop.x
        var parent1 = apop.parent1
        var parent2 = apop.parent2
        var xsite = apop.xsite
        println(f"fitness: $fitness%f x: $x%f parent1: $parent1%d parent2: $parent2%d xsite: $xsite%d ")
      }

    }
  def printStatistics(stat: bestever, gen: Int, min: Double, max: Double, avg: Double, sumfitness: Double): Unit =
    {
      println(f"Generation $gen%d Accumulated Statistics: ")

      println(f"Total Crossovers = $ncross%d, Total Mutations = $nmutation%d\n")
      println(f"min = $min%f   max = $max%f   avg = $avg%f   sum = $sumfitness%f")
      val generation = stat.generation
      println(f"Global Best Individual so far, Generation $generation%d:")
      val fitness = stat.fitness
      println(f"Fitness = $fitness%f: ")
      println("chromosome")
      stat.chrom.foreach { allel =>
        print(if (allel == false) 0 else 1)
      }
      println
    }
  def objfunc(x: Double, lchrom: Int): Double = {
    val n = 10.0
    var coef = Math.pow(2.0, lchrom) - 1.0
    coef = Math.pow(coef, n)
    Math.pow(x, n) / coef
  }

  def initPopulation(oldpop: population, objfunc: (Double, Int) => Double): Unit = {
    for (apop <- oldpop) {

      for (k <- 0 until apop.chrom.length)
        apop.chrom(k) = flip(0.5)
      var x = decode(apop.chrom)
      apop.fitness = objfunc(x, apop.chrom.length)
      apop.parent1 = 0
      apop.parent2 = 0
      apop.xsite = 0
    }
  }

  def generation(oldpop: population, newpop: population, pcross: Double, pmutation: Double, objfunc: (Double, Int) => Double): Unit =
    {
      var sumfitness: Double = 0 //summed fitness for entire population
      var jcross = 0
      var j = 0

      oldpop.foreach { apop =>
        sumfitness += apop.fitness
      }

      do {
        var mate1 = select(sumfitness, oldpop)
        var mate2 = select(sumfitness, oldpop)

        var crossoverGene = crossover(oldpop(mate1).chrom, oldpop(mate2).chrom, pcross, pmutation)

        newpop(j).chrom = crossoverGene._1
        newpop(j + 1).chrom = crossoverGene._2
        jcross = crossoverGene._3

        var x = decode(newpop(j).chrom);
        newpop(j).fitness = objfunc(x, newpop(j).chrom.length);
        newpop(j).parent1 = mate1 + 1;
        newpop(j).parent2 = mate2 + 1;
        newpop(j).xsite = jcross;

        x = decode(newpop(j + 1).chrom);
        newpop(j + 1).fitness = objfunc(x, newpop(j + 1).chrom.length);
        newpop(j + 1).parent1 = mate1 + 1;
        newpop(j + 1).parent2 = mate2 + 1;
        newpop(j + 1).xsite = jcross;

        j += j + 2;
      } while (j < (newpop.length - 1))
    }

  def select(sumfitness: Double, pop: population): Int = {
    var rand = Math.random * sumfitness;
    var partsum = 0.0
    var j = 0
    if (sumfitness != 0) {
      pop.foreach { apop =>
        if (partsum < rand)
          partsum += apop.fitness / sumfitness
        j += 1
      }
    } else
      j = rnd(0, pop.length - 1)

    (j - 1)
  }
  def crossover(parent1: chromosome, parent2: chromosome, pcross: Double, pmutation: Double): (chromosome, chromosome, Int) = {

    val lchrom = parent1.length
    var child1: chromosome = new Array[allele](lchrom)
    var child2: chromosome = new Array[allele](lchrom)
    var jcross = lchrom

    if (flip(pcross)) {
      jcross = rnd(1, lchrom - 1)
      ncross += 1
    }

    for (j <- 0 until jcross){
      child1(j) = mutation(parent1(j), pmutation)
      child2(j) = mutation(parent2(j), pmutation)
    }
    if (jcross != lchrom) {
      for (j <- jcross + 1 until lchrom) {
        child1(j) = mutation(parent2(j), pmutation)
        child2(j) = mutation(parent1(j), pmutation)
      }
    }
    (child1, child2, jcross)
  }
  def mutation(alleleval: allele, pmutation: Double): allele = {
    var mutate = flip(pmutation)
    if (mutate) {
      nmutation += 1
      !alleleval
    } else alleleval
  }
  def rnd(low: Int, high: Int): Int =
    {
      if (low >= high) low
      else {
        val i = (Math.random * (high - low + 1) + low).asInstanceOf[Int]
        if (i > high) high
        else i
      }

    }
  def flip(probability: Double): Boolean =
    if (probability == 1.0) true
    else (Math.random <= probability)

  def decode(chrom: chromosome): Double = {
    var accum = 0.0
    var powerof2 = 1.0
    chrom.foreach { allel =>
      if (allel)
        accum = accum + powerof2
      powerof2 = powerof2 * 2.0
    }
    accum
  }
  def statistics(pop: population, bestfit: bestever, gen: Int): (Double, Double, Double, Double, bestever) = {
    var sumfitness = 0.0 //summed fitness for entire population
    var max = 0.0 // maximum fitness of population
    var min = 0.0 //minimum fitness of population

    pop.foreach { apop =>
      sumfitness += apop.fitness
      if (apop.fitness > max) max = apop.fitness
      if (apop.fitness < min) min = apop.fitness

      if (apop.fitness > bestfit.fitness) {
        bestfit.chrom = apop.chrom
        bestfit.fitness = apop.fitness
        bestfit.generation = gen
      }
    }
    var avg = sumfitness / pop.length

    (min, max, avg, sumfitness, bestfit)
  }
}