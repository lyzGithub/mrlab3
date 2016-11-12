package comMain;
/**
 * Created by liyize on 16-11-2.
 */
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.HTableInterface;
import org.apache.hadoop.hbase.client.HTablePool;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.util.*;


import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

public class InvertedIndexWriteToHBase {


    private static MyHTable hb;
    private static List<Put> putList;

    /**
     * Constructor.
     *
     * @throws IOException HBase needs it.
     */
    InvertedIndexWriteToHBase() throws IOException
    {

        System.out.println("yes");
    }

    public static void main(String[] args) throws Exception {
        if(args.length != 2){
            System.out.println("wrong input args");
            System.exit(-1);
        }

        hb = new MyHTable();
        putList = new ArrayList<Put>();

        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Job job = new Job(conf, "inverted index to hbase:group st14");
        job.setJarByClass(InvertedIndexWriteToHBase.class);
        job.setMapperClass(InvertedIndexMapper.class);
        job.setCombinerClass(SumCombiner.class);
        job.setReducerClass(InvertedIndexReducer.class);
        job.setNumReduceTasks(5);//Reduce节点个数为5
        job.setPartitionerClass(NewPartitioner.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));

        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));

        if (job.waitForCompletion(true))
        {
            hb.insertDataListToTable(putList);
            hb.writeToFile();
            hb.cleanup();
            System.exit(0);
        }
        else{
            System.exit(1);
        }

    }
    /**
     * Mapper部分
     **/
    public static class InvertedIndexMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);

        /**
         输入：key：文件的当前的行的位置偏移位置，value：也就是文件具体行的内容
         输出：key：单词#文件名，value：1
         整体过程思路：取得当前的文件的名字之后，
         对value的值进行切分得出多个词语的值，将以上词语和filename进行拼接，组成key，value记1.
         */
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName().toLowerCase();
            int pos = fileName.indexOf(".");
            if (pos > 0) {
                fileName = fileName.substring(0, pos);//去后缀
            }
            Text word = new Text();
            StringTokenizer itr = new StringTokenizer(value.toString());
            for (; itr.hasMoreTokens(); ) {
                word.set(itr.nextToken() + "#" + fileName);
                context.write(word, one);//output word#filename 1
            }
        }
    }

    /**
     Combiner部分
     输入：key：单词#文件名，value：1，1，1，1，…
     输出：key：单词#文件名，value：相同的key的累加和的值sum
     整体过程思路：为了减少向reducer传输数据的总量，我们将相同的key的value的值进行累加。

     */
    public static class SumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();


        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }//combiner的reduce重载
    }

    /**
     * Partitioner部分
     输入：key：单词#文件名，value：累加和值sum
     输出：key：单词#文件名，value：累加和值sum
     整体过程和思路：为了防止将同一个单词下不同文件名的key被分到不同的reducer中，
     我们在对key进行partition操作的时候，以key中的单词做为关键词进行划分，
     以保证同一个单词被分到同一个reducer上。

     **/
    public static class NewPartitioner extends HashPartitioner<Text, IntWritable> {
        /**
         * 防止不同的term发到不同的节点，把主键临时拆开，使得partition按照term而不是（term,docid）进行分区正确选择reduce节点
         */
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            String term = key.toString().split("#")[0];//get word from word#filename word
            return super.getPartition(new Text(term), value, numReduceTasks);
        }
    }

    /**
     * Reducer部分
     输入：key：词语#文件名，value：sum1，sum2，…
     输出：key：词语\t平均出现次数，文件名：词频；文件名：词频；…

     由于输入的key都是有序的，对一个词语，每次处理信息都是保存‘文件名’和词频，
     统计总出现次数和出现的文档总数。所有消息处理完成之后，计算结果。
     **/
    public static class InvertedIndexReducer extends Reducer<Text, IntWritable, Text, Text> {
        private String term = new String();//temp for word#filename中的word
        private String last = " ";//temp for the up word
        private int countItem;//count word
        private int countDoc;//count word doc
        private StringBuilder out = new StringBuilder();//temp for the value of output
        private float average;//temp tocount averager


        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            term = key.toString().split("#")[0];//get word
            if (!term.equals(last)) {//word与上次不一样，将上次进行处理并输出
                if (!last.equals(" ")) {//比对格式，避免第一次比较时出错
                    out.setLength(out.length() - 1);//比对格式，删除value部分最后的;符号
                    average = (float) countItem / countDoc;//计算平均出现次数
                    context.write(new Text(last), new Text(String.format("%.2f,%s", average, out.toString())));//value部分拼接后输出//("%.2f,%s", f, out.toString())))
                    //context.write(new Text(last), new Text(String.format("%.2f,%s", average)));
                    //System.out.println(last +" "+String.format("%.2f,%s", average, out.toString()));
                    Put put = new Put(Bytes.toBytes(last.toString()));
                    put.addColumn(Bytes.toBytes("averageCounts"), Bytes.toBytes("averageCounts"), Bytes.toBytes(String.valueOf(average)));
                    putList.add(put);
                    countItem = 0;//初始化计算下一个word
                    countDoc = 0;
                    out = new StringBuilder();
                }
                last = term;//更新word
            }
            int sum = 0;//累加word和filename次数
            for (IntWritable val : values) {
                sum += val.get();
            }
            out.append(key.toString().split("#")[1] + ":" + sum + ";");//将filename:count; 临时存储
            countItem += sum;
            countDoc += 1;
        }

        /**
         * 上述reduce()只会在遇到新word时，处理并输出前一个word，故对于最后一个word还需要额外的处理
         * 重载cleanup()，处理最后一个word并输出
         */
        public void cleanup(Context context) throws IOException, InterruptedException {
            out.setLength(out.length() - 1);
            average = (float) countItem / countDoc;
            //context.write(new Text(last), new Text(String.format("%.2f,%s", average)));
            context.write(new Text(last), new Text(String.format("%.2f,%s", average, out.toString())));
            //System.out.println(last +" "+String.format("%.2f,%s", average, out.toString()));
            Put put = new Put(Bytes.toBytes(last.toString()));
            put.addColumn(Bytes.toBytes("averageCounts"), Bytes.toBytes("averageCounts"), Bytes.toBytes(String.valueOf(average)));
            putList.add(put);
        }
    }


}