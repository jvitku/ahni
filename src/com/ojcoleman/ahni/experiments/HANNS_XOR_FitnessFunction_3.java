package com.ojcoleman.ahni.experiments;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import org.jgapcustomised.Chromosome;

import com.anji.integration.Activator;
import com.ojcoleman.ahni.evaluation.BulkFitnessFunctionMT;
import com.ojcoleman.ahni.evaluation.HyperNEATFitnessFunction;
import com.ojcoleman.ahni.experiments.HANNS_experiments.HANNS_Experiments_Constants;
import com.ojcoleman.ahni.hyperneat.Properties;
import com.ojcoleman.ahni.nn.GridNet;

public class HANNS_XOR_FitnessFunction_3 extends HyperNEATFitnessFunction{
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
		int GRID_RANGE = 5;
//		EvaluatorHANNS evaluator = (EvaluatorHANNS)this.evaluators[threadIndex];
		int number_of_steps = 10000;
		Random r = new Random();
		int correctCount = 0;
		for (int i = 0; i < number_of_steps; i++) {
			int in1 = r.nextInt(2);
			int in2 = r.nextInt(2);
			//0->1st layer
			int or1_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][0][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][0][0]);
			int or1_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][0][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][0][1]);
			int or1_out = or(or1_1,or1_2);
			
			int or2_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][1][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][1][0]);
			int or2_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][1][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][1][1]);
			int or2_out = or(or2_1,or2_2);
			
			int and1_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][2][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][2][0]);
			int and1_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][2][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][2][1]);
			int and_1_out = and(and1_1,and1_2);
			
			int and2_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][3][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][3][0]);
			int and2_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][3][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][3][1]);
			int and2_out = and(and2_1,and2_2);
			
			int nand1_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][4][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][4][0]);
			int nand1_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][4][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][4][1]);
			int nand_1_out = and(nand1_1,nand1_2);
			
			int nand2_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][5][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][5][0]);
			int nand2_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][5][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][5][1]);
			int nand_2_out = and(nand2_1,nand2_2);
			//second part(another 6 nodes)
			int and3_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][6][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][6][0]);
			int and3_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][6][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][6][1]);
			int and3_out = and(and3_1,and3_2);
			
			int and4_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][7][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][7][0]);
			int and4_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][7][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][7][1]);
			int and4_out = and(and4_1,and4_2);
			
			int or3_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][8][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][8][0]);
			int or3_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][8][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][8][1]);
			int or3_out = or(or3_1,or3_2);
			
			int or4_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][9][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][9][0]);
			int or4_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][9][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][9][1]);
			int or4_out = or(or4_1,or4_2);
			
			int nand3_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][10][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][10][0]);
			int nand3_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][10][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][10][1]);
			int nand_3_out = and(nand3_1,nand3_2);
			
			int nand4_1 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][11][0]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][11][0]);
			int nand4_2 = (int)Math.round(in1*gridNet.getWeights()[0][0][0][0][11][1]+in2*gridNet.getWeights()[0][GRID_RANGE][1][0][11][1]);
			int nand_4_out = and(nand4_1,nand4_2);
			//1st->2nd layer
			int and5_1 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][0][0]+or2_out*gridNet.getWeights()[1][1][0][0][0][0]+
					and_1_out*gridNet.getWeights()[1][2][0][0][0][0]+and2_out*gridNet.getWeights()[1][3][0][0][0][0]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][0][0]+nand_2_out*gridNet.getWeights()[1][5][0][0][0][0]+
					and3_out*gridNet.getWeights()[1][6][0][0][0][0]+and4_out*gridNet.getWeights()[1][7][0][0][0][0]+
					or3_out*gridNet.getWeights()[1][8][0][0][0][0]+or4_out*gridNet.getWeights()[1][9][0][0][0][0]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][0][0]+nand_4_out*gridNet.getWeights()[1][11][0][0][0][0]
					);
			
			int and5_2 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][0][1]+or2_out*gridNet.getWeights()[1][1][0][0][0][1]+
					and_1_out*gridNet.getWeights()[1][2][0][0][0][1]+and2_out*gridNet.getWeights()[1][3][0][0][0][1]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][0][1]+nand_2_out*gridNet.getWeights()[1][5][0][0][0][1]+
					and3_out*gridNet.getWeights()[1][6][0][0][0][1]+and4_out*gridNet.getWeights()[1][7][0][0][0][1]+
					or3_out*gridNet.getWeights()[1][8][0][0][0][1]+or4_out*gridNet.getWeights()[1][9][0][0][0][1]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][0][1]+nand_4_out*gridNet.getWeights()[1][11][0][0][0][1]);
			int and5_out = and(and5_1,and5_2);
			
			int and6_1 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][0][0]+or2_out*gridNet.getWeights()[1][1][0][0][0][0]+
					and_1_out*gridNet.getWeights()[1][2][0][0][0][0]+and2_out*gridNet.getWeights()[1][3][0][0][0][0]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][0][0]+nand_2_out*gridNet.getWeights()[1][5][0][0][0][0]+
					and3_out*gridNet.getWeights()[1][6][0][0][0][0]+and4_out*gridNet.getWeights()[1][7][0][0][0][0]+
					or3_out*gridNet.getWeights()[1][8][0][0][0][0]+or4_out*gridNet.getWeights()[1][9][0][0][0][0]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][0][0]+nand_4_out*gridNet.getWeights()[1][11][0][0][0][0]
					);
			
			int and6_2 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][1][1]+or2_out*gridNet.getWeights()[1][1][0][0][1][1]+
					and_1_out*gridNet.getWeights()[1][2][0][0][1][1]+and2_out*gridNet.getWeights()[1][3][0][0][1][1]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][1][1]+nand_2_out*gridNet.getWeights()[1][5][0][0][1][1]+
					and3_out*gridNet.getWeights()[1][6][0][0][1][1]+and4_out*gridNet.getWeights()[1][7][0][0][1][1]+
					or3_out*gridNet.getWeights()[1][8][0][0][1][1]+or4_out*gridNet.getWeights()[1][9][0][0][1][1]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][1][1]+nand_4_out*gridNet.getWeights()[1][11][0][0][1][1]);
			int and6_out = and(and6_1,and6_2);
			
			int or5_1 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][2][0]+or2_out*gridNet.getWeights()[1][1][0][0][2][0]+
					and_1_out*gridNet.getWeights()[1][2][0][0][2][0]+and2_out*gridNet.getWeights()[1][3][0][0][2][0]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][2][0]+nand_2_out*gridNet.getWeights()[1][5][0][0][2][0]+
					and3_out*gridNet.getWeights()[1][6][0][0][2][0]+and4_out*gridNet.getWeights()[1][7][0][0][2][0]+
					or3_out*gridNet.getWeights()[1][8][0][0][2][0]+or4_out*gridNet.getWeights()[1][9][0][0][2][0]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][2][0]+nand_4_out*gridNet.getWeights()[1][11][0][0][2][0]
					);
			
			int or5_2 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][2][1]+or2_out*gridNet.getWeights()[1][1][0][0][2][1]+
					and_1_out*gridNet.getWeights()[1][2][0][0][2][1]+and2_out*gridNet.getWeights()[1][3][0][0][2][1]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][2][1]+nand_2_out*gridNet.getWeights()[1][5][0][0][2][1]+
					and3_out*gridNet.getWeights()[1][6][0][0][2][1]+and4_out*gridNet.getWeights()[1][7][0][0][2][1]+
					or3_out*gridNet.getWeights()[1][8][0][0][2][1]+or4_out*gridNet.getWeights()[1][9][0][0][2][1]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][2][1]+nand_4_out*gridNet.getWeights()[1][11][0][0][2][1]);
			int or5_out = or(or5_1,or5_2);
			
			int or6_1 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][3][0]+or2_out*gridNet.getWeights()[1][1][0][0][3][0]+
					and_1_out*gridNet.getWeights()[1][2][0][0][3][0]+and2_out*gridNet.getWeights()[1][3][0][0][3][0]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][3][0]+nand_2_out*gridNet.getWeights()[1][5][0][0][3][0]+
					and3_out*gridNet.getWeights()[1][6][0][0][3][0]+and4_out*gridNet.getWeights()[1][7][0][0][3][0]+
					or3_out*gridNet.getWeights()[1][8][0][0][3][0]+or4_out*gridNet.getWeights()[1][9][0][0][3][0]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][3][0]+nand_4_out*gridNet.getWeights()[1][11][0][0][3][0]
					);
			
			int or6_2 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][3][1]+or2_out*gridNet.getWeights()[1][1][0][0][3][1]+
					and_1_out*gridNet.getWeights()[1][2][0][0][3][1]+and2_out*gridNet.getWeights()[1][3][0][0][3][1]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][3][1]+nand_2_out*gridNet.getWeights()[1][5][0][0][3][1]+
					and3_out*gridNet.getWeights()[1][6][0][0][3][1]+and4_out*gridNet.getWeights()[1][7][0][0][3][1]+
					or3_out*gridNet.getWeights()[1][8][0][0][3][1]+or4_out*gridNet.getWeights()[1][9][0][0][3][1]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][3][1]+nand_4_out*gridNet.getWeights()[1][11][0][0][3][1]);
			int or6_out = or(or6_1,or6_2);
			
			int nand5_1 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][4][0]+or2_out*gridNet.getWeights()[1][1][0][0][4][0]+
					and_1_out*gridNet.getWeights()[1][2][0][0][4][0]+and2_out*gridNet.getWeights()[1][3][0][0][4][0]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][4][0]+nand_2_out*gridNet.getWeights()[1][5][0][0][4][0]+
					and3_out*gridNet.getWeights()[1][6][0][0][4][0]+and4_out*gridNet.getWeights()[1][7][0][0][4][0]+
					or3_out*gridNet.getWeights()[1][8][0][0][4][0]+or4_out*gridNet.getWeights()[1][9][0][0][4][0]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][4][0]+nand_4_out*gridNet.getWeights()[1][11][0][0][4][0]
					);
			
			int nand5_2 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][4][1]+or2_out*gridNet.getWeights()[1][1][0][0][4][1]+
					and_1_out*gridNet.getWeights()[1][2][0][0][4][1]+and2_out*gridNet.getWeights()[1][3][0][0][4][1]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][4][1]+nand_2_out*gridNet.getWeights()[1][5][0][0][4][1]+
					and3_out*gridNet.getWeights()[1][6][0][0][4][1]+and4_out*gridNet.getWeights()[1][7][0][0][4][1]+
					or3_out*gridNet.getWeights()[1][8][0][0][4][1]+or4_out*gridNet.getWeights()[1][9][0][0][4][1]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][4][1]+nand_4_out*gridNet.getWeights()[1][11][0][0][4][1]);
			int nand5_out = or(nand5_1,nand5_2);
			
			int nand6_1 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][5][0]+or2_out*gridNet.getWeights()[1][1][0][0][5][0]+
					and_1_out*gridNet.getWeights()[1][2][0][0][5][0]+and2_out*gridNet.getWeights()[1][3][0][0][5][0]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][5][0]+nand_2_out*gridNet.getWeights()[1][5][0][0][5][0]+
					and3_out*gridNet.getWeights()[1][6][0][0][5][0]+and4_out*gridNet.getWeights()[1][7][0][0][5][0]+
					or3_out*gridNet.getWeights()[1][8][0][0][5][0]+or4_out*gridNet.getWeights()[1][9][0][0][5][0]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][5][0]+nand_4_out*gridNet.getWeights()[1][11][0][0][5][0]
					);
			
			int nand6_2 =(int)Math.round(or1_out*gridNet.getWeights()[1][0][0][0][5][1]+or2_out*gridNet.getWeights()[1][1][0][0][5][1]+
					and_1_out*gridNet.getWeights()[1][2][0][0][5][1]+and2_out*gridNet.getWeights()[1][3][0][0][5][1]+
					nand_1_out*gridNet.getWeights()[1][4][0][0][5][1]+nand_2_out*gridNet.getWeights()[1][5][0][0][5][1]+
					and3_out*gridNet.getWeights()[1][6][0][0][5][1]+and4_out*gridNet.getWeights()[1][7][0][0][5][1]+
					or3_out*gridNet.getWeights()[1][8][0][0][5][1]+or4_out*gridNet.getWeights()[1][9][0][0][5][1]+
					nand_3_out*gridNet.getWeights()[1][10][0][0][5][1]+nand_4_out*gridNet.getWeights()[1][11][0][0][5][1]);
			int nand6_out = or(nand6_1,nand6_2);
			//2ndlayer->3rd layer
			int out =(int)Math.round(and5_out*gridNet.getWeights()[2][0][0][0][0][0]+and6_out*gridNet.getWeights()[2][1][0][0][1][0]+
					or5_out*gridNet.getWeights()[2][2][0][0][2][0]+or6_out*gridNet.getWeights()[2][0][0][0][3][0]
					+nand5_out*gridNet.getWeights()[2][1][0][0][4][0]+nand6_out*gridNet.getWeights()[2][2][0][0][5][0]);
			
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

	public int nand(int a, int b){
		return (a + b == 2) ? 0 : 1;
	}
	
	public int and(int a, int b){
		return (a + b == 2) ? 0 : 1;
	}
	
	public int or(int a, int b){
		return (a + b == 2) ? 1 : 0;
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
