package com.ojcoleman.ahni.experiments;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.jgapcustomised.Chromosome;

import ca.nengo.model.StructuralException;

import com.anji.integration.Activator;
import com.ojcoleman.ahni.evaluation.BulkFitnessFunctionMT;
import com.ojcoleman.ahni.evaluation.HyperNEATFitnessFunction;
import com.ojcoleman.ahni.experiments.HANNS_experiments.HANNS_Experiments_Constants;
import com.ojcoleman.ahni.hyperneat.Properties;
import com.ojcoleman.ahni.nn.GridNet;

import ctu.nengorosHeadless.network.connections.InterLayerWeights;
import ctu.nengorosHeadless.network.connections.impl.IOGroup;
import design.models.QLambdaTestSim;

public class QLambdaFitness14VectorSchool extends HyperNEATFitnessFunction{
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
		Float[] vector = new Float[14];
//		vector[0] = 0.0f;
//		vector[1] = 1.0f;
//		vector[2] = 0.0f;
//		vector[3] = 0.0f;
//		vector[4] = 1.0f;
//		vector[5] = 0.0f;
//		vector[6] = 0.0f;
//		vector[7] = 0.0f;
//		vector[8] = 0.0f;
//		vector[9] = 1.0f;
//		vector[10] = 0.0f;
//		vector[11] = 0.0f;
//		vector[12] = 0.0f;
//		vector[13] = 1.0f;
		int GRID_SIZE = gridNet.getWeights()[0].length;
		int R_POS = 0;
		int MOT_IM_POS = GRID_SIZE/4;
		int X_POS = (GRID_SIZE/4)*2;
		int Y_POS = (GRID_SIZE/4)*3;
		
		vector[0] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][MOT_IM_POS][MOT_IM_POS];
		vector[1] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][R_POS][R_POS];
		vector[2] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][X_POS][X_POS];
		vector[3] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][Y_POS][Y_POS];
		vector[4] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][MOT_IM_POS][MOT_IM_POS];
		vector[5] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][R_POS][R_POS];
		vector[6] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][X_POS][X_POS];
		vector[7] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][Y_POS][Y_POS];
		vector[8] = (float)gridNet.getWeights()[0][X_POS][X_POS][0][MOT_IM_POS][MOT_IM_POS];
		vector[9] = (float)gridNet.getWeights()[0][X_POS][X_POS][0][X_POS][X_POS];
		vector[10] = (float)gridNet.getWeights()[0][X_POS][X_POS][0][Y_POS][Y_POS];
		vector[11] = (float)gridNet.getWeights()[0][Y_POS][Y_POS][0][MOT_IM_POS][MOT_IM_POS];
		vector[12] = (float)gridNet.getWeights()[0][Y_POS][Y_POS][0][X_POS][X_POS];
		vector[13] = (float)gridNet.getWeights()[0][Y_POS][Y_POS][0][Y_POS][Y_POS];
		Float[] genome = QLambdaTestSim.decode(vector);
		try {
			evaluator.getSimulator().setInitWeights();
			evaluator.getSimulator().getInterLayerNo(0).setVector(genome);
		} catch (StructuralException e) {
			e.printStackTrace();
		}
		
		double fitnessVal = evaluator.evaluateGenomeInSimulator();
		genotype.setPerformanceValue(fitnessVal);
		genotype.setFitnessValue(fitnessVal);
		if(BulkFitnessFunctionMT.getBestPerformingActivatorPerformance() < fitnessVal){
			InterLayerWeights[] weights = new InterLayerWeights[1];
			weights[0] = evaluator.getSimulator().getInterLayerNo(0);
			BulkFitnessFunctionMT.setBestPerformingActivator(weights);
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
			InterLayerWeights weights = getBestPerformingActivator()[0];
			setBestPerformingActivator(null);
			double perf = getBestPerformingActivatorPerformance();
			setBestPerformingActivatorPerformance(0.0f);
			
			ArrayList<IOGroup> inputs = weights.getInputs();
			ArrayList<IOGroup> outputs = weights.getOutputs();
			writer.println("Best performing HANNS weights with performance"+perf+":");
			writer.println(BulkFitnessFunctionMT.getBestPerformingSpecie());
			for (int i = 0; i < inputs.size(); i++) {
				for (int j = 0; j < outputs.size(); j++) {
					writer.println("Weights between input group "+i+" and output group "+j+" :");
					// this submatrix of the weightMatrix defines connections only between these two 
					float[][] submatrix = null;
					try {
						submatrix = weights.getWeightsBetween(i, j);
					} catch (StructuralException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					for (int k = 0; k < submatrix.length; k++) {
						for (int k2 = 0; k2 < submatrix[0].length; k2++) {
							writer.println("Weight between "+k+" , "+k2+" = "+submatrix[k][k2]);
						}
					}
					writer.println("-------------------------------------------------------------------");
				}
			}
			writer.println();
			writer.close();
		}
		catch(IOException e){
			System.err.println(e.getMessage()+e.getStackTrace());
		}
	}
}
