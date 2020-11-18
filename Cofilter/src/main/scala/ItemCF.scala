import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

object ItemCF {

  case class UserItemVote(userId:String, itemId:String, vote:Float)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .getOrCreate()

    val userItemData = spark.read.format("csv")
      .option("header", true)
      .option("delimiter", ",")
      .option("inferSchema", true)//自动推断数据类型
      .schema(ScalaReflection.schemaFor[UserItemVote].dataType.asInstanceOf[StructType])
      .load("C://Users/mirac/Desktop/cf_item_based.csv")

    //userItemData.show(false)

    // 用户-物品 打分表
    val userItemTmp = userItemData.groupBy(col("userId"))
      .pivot("itemId")//透视函数，行转列，只能跟在groupBy后面
      .sum("vote")
    //userItemTmp.show(false)

    // 计算物品相似度矩阵(共现矩阵)
    val rddTmp = userItemData.rdd.filter(x => x.getFloat(2) > 0)
      .map(x => {(x.getString(0), x.getString(1))})
    import spark.implicits._
    // (userId, (itemId1, itemId2))
    val itemSim = rddTmp.join(rddTmp)
      .map(x => (x._2, 1))
      .reduceByKey(_+_)
      .filter(x => x._1._1 != x._1._2)
      .map(x => {
        (x._1._1, x._1._2, x._2)
      })
      .toDF("itemId_1", "itemId_2", "sim")

    //itemSim.show(false)

    val itemSimTmp = itemSim.groupBy("itemId_1")
      .pivot("itemId_2")
      .sum("sim")
      .withColumnRenamed("itemId_1", "itemId")
        .na.fill(0)
    //itemSimTmp.show(false)

    // 用户-物品打分矩阵 * 物品相似度矩阵 = 推荐列表
    val itemInterest = userItemData.join(
      itemSim,
      itemSim("itemId_2") === userItemData("itemId")
    ).where(col("userId") === "User_A")
      .select(
        userItemData("userId"),
        userItemData("itemId"),
        userItemData("vote"),
        itemSim("itemId_1"),
        itemSim("itemId_2"),
        itemSim("sim"),
        (userItemData("vote") * itemSim("sim")).as("interest")
      )

    //itemInterest.show(false)
    val interestList = itemInterest.groupBy(col("itemId_1")).agg(sum(col("interest")))
      .withColumnRenamed("sum(interest)", "interest")
      .orderBy(desc("interest"))

    interestList.show(false)
  }

}
