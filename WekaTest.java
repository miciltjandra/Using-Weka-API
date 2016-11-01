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
	private static Instances train;
	private static NaiveBayes classifier;
	private static Evaluation eval;

	public static void load(String filename) throws Exception {
		ArffLoader loader = new ArffLoader();
	    loader.setFile(new File(filename));
	    train = loader.getDataSet();
	    train.setClassIndex(train.numAttributes() - 1);
	}

	public static void discretize() throws Exception {
		//setup discretize filter
		Discretize filter = new Discretize();
	    filter.setInputFormat(train);

	    //apply discretize
		Instances filtered = Filter.useFilter(train, filter);
	    train = filtered;
	}

	public static void naiveBayes() throws Exception {
		//train NaiveBayes
		classifier = new NaiveBayes();
		classifier.buildClassifier(train);
	}

	public static void crossValidate() throws Exception {
		eval = new Evaluation(train);
	    eval.crossValidateModel(classifier, train, 10, new Random(1));
	}

	public static void printEvalResult() throws Exception {
		System.out.println(classifier);
		System.out.println(eval.toSummaryString());
 		System.out.println(eval.toClassDetailsString());
 		System.out.println(eval.toMatrixString());
	}

	public static void main(String args[]) throws Exception {
		System.out.print("Please input path to dataset : ");
		Scanner sc = new Scanner(System.in);
		String filename = sc.next();
		//System.out.println(filename + "aa");

		// load data
	    load(filename);

	    //setup discretize filter
	    discretize();


	    // train NaiveBayes
	    //NaiveBayes nb = new NaiveBayes();
	    //nb.buildClassifier(train);
	    naiveBayes();
	    //System.out.println(classifier);
	    

	    crossValidate();
	    //eval.evaluateModel(nb, structure);
	    printEvalResult();
 		
	}
}
