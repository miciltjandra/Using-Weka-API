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

	public static void fulltraining() throws Exception {
		eval = new Evaluation(train);
		eval.evaluateModel(classifier, train);
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
	
	public static void saveModel(String modelname) throws Exception {
		String outname = "";
		outname = outname.concat(modelname);
		outname = outname.concat(".model");
		weka.core.SerializationHelper.write(outname, train);
	}
	
	/*
	public static void loadModel(String modelfile) throws Exception {
		NaiveBayes classifier = (NaiveBayes) weka.core.SerializationHelper.read(modelfile);
	}
	*/
	

	public static void main(String args[]) throws Exception {
		System.out.print("Please input path to dataset : ");
		Scanner sc = new Scanner(System.in);
		String filename = sc.next();
		//System.out.println(filename + "aa");

		// load data
	    load(filename);

		
	    //setup filter
		System.out.println("Filter 0. No 1. Discretisize 2. NumericToNominal");
		System.out.print("Use filter : ");
		String fvar = sc.next();
		if ("1".equals(fvar)) {
			discretize();
		} 
		else if ("2".equals(fvar)) {
			//numericToNominal();
		}

	    // train NaiveBayes
	    //NaiveBayes nb = new NaiveBayes();
	    //nb.buildClassifier(train);
	    naiveBayes();
	    //System.out.println(classifier);
	    
		//use schema
		System.out.println("Schema 1. 10-fold Cross Validate 2. Full Training");
		System.out.print("Use schema : ");
		String svar = sc.next();
		if ("1".equals(svar)) {
			crossValidate();
		} 
		else if ("2".equals(svar)) {
			fulltraining();
		}

	    //eval.evaluateModel(nb, structure);
	    printEvalResult();
		
		//save model
		System.out.print("Want to save model?(Y/N) : ");
		String savevar = sc.next();
		if ("Y".equals(savevar)) {
			System.out.print("model name : ");
			String modelname = sc.next();
			saveModel(modelname);
			System.out.println("model saved");
		}
 		
	}
}
