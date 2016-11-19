package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.functions._


case class Song(year: Double, f1: Double,  f2: Double,  f3: Double)

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("raw")

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rdd.take(5).foreach(println) 
    
    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(x => x.split(','))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(s => Song(s(0).toDouble, s(1).toDouble, s(2).toDouble, s(3).toDouble))

    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF()
    songsDf.createOrReplaceTempView("songs")
    
    println("How many songs there are in the DataFrame?")
    println(songsDf.count())
    sqlContext.sql("select count(*) from songs").show()
    
    
        println("How many songs were released between the years 1998 and 2000?")
    println(songsDf.filter($"year" >= 1998 and $"year" <= 2000).count())
    sqlContext.sql("select count(*) from songs where year between 1998 and 2000").show()
    
    
    
        println("What is the min, max and mean value of the year column?")
    //songsDf.select(min($"year")).rdd.first.getDouble(0)
    songsDf.groupBy().max("year").show()
    songsDf.groupBy().mean("year").show()
    songsDf.groupBy().min("year").show()
    sqlContext.sql("select min(year) , max(year) ,avg(year) from songs").show()
    
    
    
        println("Show the number of songs per year between the years 2000 and 2010?")
    songsDf.filter($"year" >= 2000 and $"year" <= 2010).groupBy("year").count().show()
    sqlContext.sql("select year, count(*) from songs s where s.year between 2000 and 2010 group by year").show()
    
    
  }
}