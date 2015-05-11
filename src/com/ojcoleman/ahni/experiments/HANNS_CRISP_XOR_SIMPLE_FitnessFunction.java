package com.ojcoleman.ahni.experiments;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import org.jgapcustomised.Chromosome;

import ca.nengo.model.StructuralException;

import com.anji.integration.Activator;
import com.ojcoleman.ahni.evaluation.BulkFitnessFunctionMT;
import com.ojcoleman.ahni.evaluation.HyperNEATFitnessFunction;
import com.ojcoleman.ahni.experiments.HANNS_experiments.HANNS_Experiments_Constants;
import com.ojcoleman.ahni.hyperneat.Properties;
import com.ojcoleman.ahni.nn.GridNet;

import ctu.nengorosHeadless.network.connections.impl.IOGroup;
import design.models.QLambdaTestSim;

public class HANNS_CRISP_XOR_SIMPLE_FitnessFunction extends HyperNEATFitnessFunction{
	private final static int DISTANCE_BETWEEN_IO_GROUPS = 0;
	private String activatorLogFilePath;
	
	/**
	 * See <a href=" {@docRoot} /params.htm" target="anji_params">Parameter Details </a> for specific property settings.
	 * 
	 * @param props configuration parameters
	 */
	public void init(Properties props) {
		super.init(props);
		activatorLogFilePath = props.getProperty(HANNS_Experiments_Constants.ACTIVATOR_LOG_FILE_PATH);
		File f = new File(activatorLogFilePath);
		f.delete();
		f.getParentFile().mkdirs();
	}


	public void initialiseEvaluation() {
	}
	
	@Override
	public void finaliseEvaluation(){
		printActivatorToFile();
	}

	/**
	 * Evaluate given individual by presenting the stimuli to the network in a random order to ensure the underlying
	 * network is not memorising the sequence of inputs. Calculation of the fitness based on error is delegated to the
	 * subclass. This method adjusts fitness for network size, based on configuration.
	 */
	protected double evaluate(Chromosome genotype, Activator activator, int threadIndex) {
		GridNet gridNet = (GridNet)activator;
		EvaluatorHANNS evaluator = (EvaluatorHANNS)this.evaluators[threadIndex];
		
		int LAYER_COUNT = gridNet.getWeights().length;
		for (int n = 0; n < LAYER_COUNT; n++) {
			ArrayList<IOGroup> inputs = evaluator.getSimulator().getInterLayerNo(n).getInputs();
			ArrayList<IOGroup> outputs = evaluator.getSimulator().getInterLayerNo(n).getOutputs();
			for (int i = 0; i < inputs.size(); i++) {
				for (int j = 0; j < outputs.size(); j++) {
					int inStartInd = inputs.get(i).getStartingIndex();
					int inNoUnits = inputs.get(i).getNoUnits();
					
					int outStartInd = outputs.get(j).getStartingIndex();
					int outNoUnits = outputs.get(j).getNoUnits();
					// this submatrix of the weightMatrix defines connections only between these two 
					float[][] submatrix = new float[inNoUnits][outNoUnits];
					for (int k = 0; k < submatrix.length; k++) {
						for (int k2 = 0; k2 < submatrix[0].length; k2++) {
//							submatrix[k][k2] = (float)gridNet.getWeights()[n][i][k][0][j][k2];
							submatrix[k][k2] = (float)gridNet.getWeights()[n][0][inStartInd+k][0][0][outStartInd+k2];
	//						if((float)gridNet.getWeights()[0][0][inStartInd+k+i*DISTANCE_BETWEEN_IO_GROUPS][0][0][outStartInd+k2+j*DISTANCE_BETWEEN_IO_GROUPS] > 0.5){
	//							submatrix[k][k2] = 1.0f;
	//						}
	//						else{
	//							submatrix[k][k2] = 0.0f;
	//						}
						}
					}
					try{
						evaluator.getSimulator().getInterLayerNo(n).setWeightsBetween(i, j, submatrix);
					}
					catch (StructuralException e) {
						e.printStackTrace();
						System.err.println("Connection weights not set");
						return 0.0f;
					}	
				}
			}
		}
		
		
		double fitnessVal = (double)evaluator.evaluateGenomeInSimulator();//(double)1000000;
		if(fitnessVal < 0){
			fitnessVal = 0;
		}
		genotype.setPerformanceValue(fitnessVal);
		genotype.setFitnessValue(fitnessVal);
		if(BulkFitnessFunctionMT.getBestPerformingActivatorPerformance() < fitnessVal){
			int vector_size = 0;
			for (int i = 0; i < LAYER_COUNT; i++) {
				ArrayList<IOGroup> inputs = evaluator.getSimulator().getInterLayerNo(i).getInputs();
				int inputs_count = 0;
				for (int j = 0; j < inputs.size(); j++) {
					inputs_count += inputs.get(j).getNoUnits();
				}
				ArrayList<IOGroup> outputs = evaluator.getSimulator().getInterLayerNo(i).getOutputs();
				int outputs_count = 0;
				for (int j = 0; j < outputs.size(); j++) {
					outputs_count += outputs.get(j).getNoUnits();
				}
				vector_size += inputs_count*outputs_count;
			}
			Float[] vector = new Float[vector_size];
			int pos = 0;
			for (int i = 0; i < LAYER_COUNT; i++) {
				float[] other = evaluator.getSimulator().getInterLayerNo(i).getVector();
				for (int j = 0; j < other.length; j++) {
					vector[pos++] = other[j];
				}
			}
			BulkFitnessFunctionMT.setBestPerformingActivator(vector);
			BulkFitnessFunctionMT.setBestPerformingActivatorPerformance((float)fitnessVal);
			if(genotype.getSpecie() != null){
				BulkFitnessFunctionMT.setBestPerformingSpecie(genotype.getSpecie().getID(), genotype.getSpecie().getAge(), genotype.getSpecie().size());
			}
		}
		return fitnessVal;
	}

	private void printActivatorToFile(){
		if(getBestPerformingActivator() == null){
			return;
		}
		try{
			PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(activatorLogFilePath, true)));
//			InterLayerWeights weights = getBestPerformingActivator()[0
//			setBestPerformingActivator(null);
			double perf = getBestPerformingActivatorPerformance();
			setBestPerformingActivatorPerformance(0.0f);
			
//			ArrayList<IOGroup> inputs = weights.getInputs();
//			ArrayList<IOGroup> outputs = weights.getOutputs();
			writer.println("Best performing HANNS weights with performance"+perf+":");
			writer.println(BulkFitnessFunctionMT.getBestPerformingSpecie());
			writer.println(Arrays.toString(getBestPerformingActivator()));
//			for (int i = 0; i < inputs.size(); i++) {
//				for (int j = 0; j < outputs.size(); j++) {
//					writer.println("Weights between input group "+i+" and output group "+j+" :");
//					// this submatrix of the weightMatrix defines connections only between these two 
//					float[][] submatrix = null;
//					try {
//						submatrix = weights.getWeightsBetween(i, j);
//					} catch (StructuralException e) {
//						// TODO Auto-generated catch block
//						e.printStackTrace();
//					}
//					for (int k = 0; k < submatrix.length; k++) {
//						for (int k2 = 0; k2 < submatrix[0].length; k2++) {
//							writer.println("Weight between "+k+" , "+k2+" = "+submatrix[k][k2]);
//						}
//					}
//					writer.println("-------------------------------------------------------------------");
//				}
//			}
			writer.println();
			writer.close();
		}
		catch(IOException e){
			System.err.println(e.getMessage()+e.getStackTrace());
		}
	}
}
