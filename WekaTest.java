import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.io.File;
import java.util.*;

import weka.core.converters.ConverterUtils.DataSource;

public class WekaTest {
	public static void main(String args[]) throws Exception {
		System.out.println("a");
		Scanner sc = new Scanner(System.in);
		String filename = sc.next();
		System.out.println(filename + "aa");

		// load data
	    ArffLoader loader = new ArffLoader();
	    loader.setFile(new File(filename));
	    Instances structure = loader.getDataSet();
	    structure.setClassIndex(structure.numAttributes() - 1);

	    //setup discretize filter
	    Discretize filter = new Discretize();
	    filter.setInputFormat(structure);

	    //apply discretize
	    //Instances filtered = Filter.useFilter(structure, filter);
	    Instances filtered = structure;


	    // train NaiveBayes
	    NaiveBayes nb = new NaiveBayes();
	    nb.buildClassifier(filtered);
	    System.out.println(nb);

	    Evaluation eval = new Evaluation(filtered);
	    eval.crossValidateModel(nb, filtered, 10, new Random(1));
	    //eval.evaluateModel(nb, structure);
 		System.out.println(eval.toSummaryString());
 		System.out.println(eval.toClassDetailsString());
 		System.out.println(eval.toMatrixString());
	}
}