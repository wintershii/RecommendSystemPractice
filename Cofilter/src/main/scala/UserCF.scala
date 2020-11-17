import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

object UserCF {
  case class UserItem(userId:String, itemId:String, score:Double)

  /**
  基于用户的协同过滤有以下几个缺点：
    1. 如果用户量很大，计算量就很大
    2. 用户-物品打分矩阵是一个非常非常非常稀疏的矩阵，会面临大量的null值
    很难得到两个用户的相似性
    3. 会将整个用户-物品打分矩阵都载入到内存，而往往这个用户-物品打分矩阵是一个
    非常大的矩阵

    所以通常不太建议使用基于用户的协同过滤
    **/

  def main(args: Array[String]): Unit = {
    //导入数据
    val spark = SparkSession.builder()
      .master("local")
      .getOrCreate()
    val df = spark.read.format("csv")
      .option("header", true)
      .option("delimiter", ",")
      .option("inferschema", true)
      .schema(ScalaReflection.schemaFor[UserItem].dataType.asInstanceOf[StructType])
      .load("C://Users/mirac/Desktop/cf_user_based.csv")

    df.show(false)
    df.printSchema()

    // 通过余弦相似度计算用户相似度
    // 余弦相似度的公式 ： (A * B) / (|A| * |B|)
    // 分子：向量各个维度的乘积和；分母：每个向量模的成绩
    import spark.implicits._
    // 分母
    val dfScoreMod = df.rdd.map(x => (x(0).toString, x(2).toString))
      .groupByKey()// 按照用户id进行分组
      .mapValues(score=>math.sqrt(
        score.toArray.map(
          itemScore=>math.pow(itemScore.toDouble, 2)
        ).reduce(_+_)
      )).toDF("userId", "mod")
    dfScoreMod.show(false)
    // 分子

    val _dfTemp = df.select(
      col("userId").as("_userId"),
      col("itemId"),
      col("score").as("_score")
    )

    //把两两用户都放到同一张表里
    val _df = df.join(_dfTemp, df("itemId") === _dfTemp("itemId"))
      .where(
        df("userId") =!= _dfTemp("_userId")
      )
      .select(
        df("itemId"),
        df("userid"),
        _dfTemp("_userId"),
        df("score"),
        _dfTemp("_score")
      )
    _df.show(false)

    //两两向量的维度乘积总和
    val df_mol = _df.select(
      col("userId"),
      col("_userId"),
      (col("score") * col("_score")).as("score_mol")
    ).groupBy(col("userId"), col("_userId"))
      .agg(sum("score_mol"))
      .withColumnRenamed("sum(score_mol)", "mol")

    df_mol.show(false)

    // 计算两两向量的余弦相似度
    val _dfScoreMod = dfScoreMod.select(
      col("userId").as("_userId"),
      col("mod").as("_mod")
    )

    val sim = df_mol.join(dfScoreMod, df_mol("userId") === dfScoreMod("userId"))
      .join(_dfScoreMod, df_mol("_userId") === _dfScoreMod("_userId"))
      .select(df_mol("userId"), df_mol("_userId"), df_mol("mol"), dfScoreMod("mod"), _dfScoreMod("_mod"))

    sim.show(false)

    // 求出 分子/分母（对于每个用户以及其向量用户）
    val cosSim = sim.select(
      col("userId"),
      col("_userId"),
      (col("mol") / (col("mod") * col("_mod"))).as("cos_sim")
    )

    cosSim.show(false)


    // 列出某个用户的topN相似用户
    val topN = cosSim.rdd.map(x=>(
      (x(0).toString,
        (x(1).toString, x(2).toString)
      )
    )).groupByKey()
      .mapValues(_.toArray.sortWith((x, y) => x._2 > y._2))
      .flatMapValues(x => x)
      .toDF("userId", "sim_sort")
      .select(
        col("userId"),
        col("sim_sort._1").as("_userId"),
        col("sim_sort._2").as("cos_sim")
      ).where(col("userId") === "1")

    topN.show(false)
  }

}
