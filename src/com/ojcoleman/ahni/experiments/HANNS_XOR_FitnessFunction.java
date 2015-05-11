package com.ojcoleman.ahni.experiments;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import org.jgapcustomised.Chromosome;

import ca.nengo.model.StructuralException;

import com.anji.integration.Activator;
import com.ojcoleman.ahni.evaluation.BulkFitnessFunctionMT;
import com.ojcoleman.ahni.evaluation.HyperNEATFitnessFunction;
import com.ojcoleman.ahni.experiments.HANNS_experiments.HANNS_Experiments_Constants;
import com.ojcoleman.ahni.hyperneat.Properties;
import com.ojcoleman.ahni.nn.GridNet;

import design.models.QLambdaTestSim;

public class HANNS_XOR_FitnessFunction extends HyperNEATFitnessFunction{
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
//		printActivatorToFile();
	}

	/**
	 * Evaluate given individual by presenting the stimuli to the network in a random order to ensure the underlying
	 * network is not memorising the sequence of inputs. Calculation of the fitness based on error is delegated to the
	 * subclass. This method adjusts fitness for network size, based on configuration.
	 */
	protected double evaluate(Chromosome genotype, Activator activator, int threadIndex) {
		GridNet gridNet = (GridNet)activator;
//		EvaluatorHANNS evaluator = (EvaluatorHANNS)this.evaluators[threadIndex];
		int number_of_steps = 10000;
		Random r = new Random();
		int correctCount = 0;
		for (int i = 0; i < number_of_steps; i++) {
			int in1 = r.nextInt(2);
			int in2 = r.nextInt(2);
			int or1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][0][0]+in2*gridNet.getWeights()[0][1][1][0][0][0]);
			int or2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][0][1]+in2*gridNet.getWeights()[0][1][1][0][0][1]);
			int or_out = (or1 + or2 == 0) ? 0 : 1;
			int nand1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][1][0]+in2*gridNet.getWeights()[0][1][1][0][1][0]);
			int nand2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][1][1]+in2*gridNet.getWeights()[0][1][1][0][1][1]);
			int nand_out = (nand1 + nand2 == 2) ? 0 : 1;
			int and1 = (int)Math.round(or_out*gridNet.getWeights()[1][0][0][0][0][0]+nand_out*gridNet.getWeights()[1][1][1][0][0][0]);
			int and2 = (int)Math.round(or_out*gridNet.getWeights()[1][0][0][0][1][1]+nand_out*gridNet.getWeights()[1][1][1][0][1][1]);
			int out = (and1 + and2 == 2) ? 1 : 0;
			if((in1 == 1 & in2 == 0) | (in1 == 0 & in2 == 1)){
				if(out == 1){
					correctCount++;
				}
			}
			else{
				if(out == 0){
					correctCount++;
				}
			}
		}
		double val = (double)correctCount/(double)number_of_steps;
		genotype.setPerformanceValue(val);
		genotype.setFitnessValue(val);
		return val;
//		Float[] vector = new Float[14];
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
//		int GRID_SIZE = gridNet.getWeights()[0].length;
//		int R_POS = 0;
//		int MOT_IM_POS = GRID_SIZE/4;
//		int X_POS = (GRID_SIZE/4)*2;
//		int Y_POS = (GRID_SIZE/4)*3;
//		
//		vector[0] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][MOT_IM_POS][MOT_IM_POS];
//		vector[1] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][R_POS][R_POS];
//		vector[2] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][X_POS][X_POS];
//		vector[3] = (float)gridNet.getWeights()[0][R_POS][R_POS][0][Y_POS][Y_POS];
//		vector[4] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][MOT_IM_POS][MOT_IM_POS];
//		vector[5] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][R_POS][R_POS];
//		vector[6] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][X_POS][X_POS];
//		vector[7] = (float)gridNet.getWeights()[0][MOT_IM_POS][MOT_IM_POS][0][Y_POS][Y_POS];
//		vector[8] = (float)gridNet.getWeights()[0][X_POS][X_POS][0][MOT_IM_POS][MOT_IM_POS];
//		vector[9] = (float)gridNet.getWeights()[0][X_POS][X_POS][0][X_POS][X_POS];
//		vector[10] = (float)gridNet.getWeights()[0][X_POS][X_POS][0][Y_POS][Y_POS];
//		vector[11] = (float)gridNet.getWeights()[0][Y_POS][Y_POS][0][MOT_IM_POS][MOT_IM_POS];
//		vector[12] = (float)gridNet.getWeights()[0][Y_POS][Y_POS][0][X_POS][X_POS];
//		vector[13] = (float)gridNet.getWeights()[0][Y_POS][Y_POS][0][Y_POS][Y_POS];
//		Float[] genome = QLambdaTestSim.decode(vector);
//		try {
//			evaluator.getSimulator().setInitWeights();
//			evaluator.getSimulator().getInterLayerNo(0).setVector(genome);
//		} catch (StructuralException e) {
//			e.printStackTrace();
//		}
//		
//		double fitnessVal = evaluator.evaluateGenomeInSimulator();
//		genotype.setPerformanceValue(fitnessVal);
//		genotype.setFitnessValue(fitnessVal);
//		if(BulkFitnessFunctionMT.getBestPerformingActivatorPerformance() < fitnessVal){
////			InterLayerWeights[] weights = new InterLayerWeights[1];
////			weights[0] = evaluator.getSimulator().getInterLayerNo(0);
//			BulkFitnessFunctionMT.setBestPerformingActivator(vector);
//			BulkFitnessFunctionMT.setBestPerformingActivatorPerformance((float)fitnessVal);
//			if(genotype.getSpecie() != null){
//				BulkFitnessFunctionMT.setBestPerformingSpecie(genotype.getSpecie().getID(), genotype.getSpecie().getAge(), genotype.getSpecie().size());
//			}
//		}
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
