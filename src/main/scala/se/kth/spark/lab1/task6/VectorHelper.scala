package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
  var a1:Array[Double]  = v1.toArray
  val a2:Array[Double]  = v2.toArray
  var sum = 0.0
    for( i <- 0 to a1.length -1 )
   {
    sum =  sum + (a1(i) * a2(i))
   }
    return sum
  }

  def dot(v: Vector, s: Double): Vector = {
    var a:Array[Double]  = v.toArray
    for( i <- 0 to a.length -1 )
   {
      a(i)  *= s
    }
    return Vectors.dense(a)
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    var a1:Array[Double]  = v1.toArray
    val a2:Array[Double]  = v2.toArray
    for( i <- 0 to a1.length -1 )
   {
    a1(i) += a2(i)
   }
    return Vectors.dense(a1)
  }
  

  def fill(size: Int, fillVal: Double): Vector = {
    val array:Array[Double] = Array.fill[Double](size)(fillVal)
    return Vectors.dense(array)
  }
  
  
  
  def main(args: Array[String]) {
  
    println(dot (dot(sum(fill(10,7), fill(10,3)), 2) , dot(sum(fill(10,7), fill(10,3))  , 2))    )
    
    
    
    val array: Array[Double] = Array(1.9, 2.9, 3.4, 3.5)
    var a:Vector = Vectors.dense(array)
      for( i <- 0 to a.size -1 )
   {
     println(a(i))
   }
    
    
    var b:Vector = Vectors.dense(1.0, 2.0, 3.0)
       // a(i) = fillVal
 
    
  }
  
  
}