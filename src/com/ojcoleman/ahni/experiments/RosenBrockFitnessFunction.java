package com.ojcoleman.ahni.experiments;

import org.jgapcustomised.Chromosome;

import com.anji.integration.Activator;
import com.ojcoleman.ahni.evaluation.BulkFitnessFunctionMT;
import com.ojcoleman.ahni.evaluation.HyperNEATFitnessFunction;
import com.ojcoleman.ahni.hyperneat.Properties;
import com.ojcoleman.ahni.nn.BainNN;
import com.ojcoleman.ahni.nn.GridNet;
import com.ojcoleman.ahni.transcriber.HyperNEATTranscriber;

import design.ea.tasks.Rosenbrock;

public class RosenBrockFitnessFunction extends HyperNEATFitnessFunction {

	public void init(Properties props) {
		super.init(props);
	}

	public void initialiseEvaluation() {
		
	}
	
	public void finaliseEvaluation(){
		System.out.println(BulkFitnessFunctionMT.getBestPerformingActivatorPerformance());
	}

	public static final double MAX_WEIGHT = 10;
	
	
	/**
	 * Evaluate given individual by presenting the stimuli to the network in a random order to ensure the underlying
	 * network is not memorising the sequence of inputs. Calculation of the fitness based on error is delegated to the
	 * subclass. This method adjusts fitness for network size, based on configuration.
	 */
	protected double evaluate(Chromosome genotype, Activator activator, int threadIndex) {
		GridNet substrate = (GridNet)activator;
		Double[] max_vec = {-MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT,
				-MAX_WEIGHT, -MAX_WEIGHT, -MAX_WEIGHT};
		Double[] vec = {substrate.getWeights()[0][0][0][0][0][0],substrate.getWeights()[0][0][0][0][1][1],substrate.getWeights()[0][0][0][0][2][2],
				substrate.getWeights()[0][0][0][0][3][3],substrate.getWeights()[0][1][1][0][0][0],substrate.getWeights()[0][1][1][0][1][1],
				substrate.getWeights()[0][1][1][0][2][2],substrate.getWeights()[0][1][1][0][3][3],substrate.getWeights()[0][2][2][0][1][1],
				substrate.getWeights()[0][2][2][0][2][2],substrate.getWeights()[0][2][2][0][3][3],substrate.getWeights()[0][3][3][0][1][1],
				substrate.getWeights()[0][3][3][0][2][2],substrate.getWeights()[0][3][3][0][3][3]};
		double max = eval(max_vec);
		double val = eval(vec);
		val = Math.abs((val/max)-1);
		if(val > BulkFitnessFunctionMT.getBestPerformingActivatorPerformance()){
			BulkFitnessFunctionMT.setBestPerformingActivatorPerformance(val);
		}
		if(val == 1){
			System.out.println(substrate.getWeights()[0][0][0][0][0][0]+","+substrate.getWeights()[0][0][0][0][1][1]+","+substrate.getWeights()[0][0][0][0][2][2]+","+
					substrate.getWeights()[0][0][0][0][3][3]+","+substrate.getWeights()[0][1][1][0][0][0]+","+substrate.getWeights()[0][1][1][0][1][1]+","+
					substrate.getWeights()[0][1][1][0][2][2]+","+substrate.getWeights()[0][1][1][0][3][3]+","+substrate.getWeights()[0][2][2][0][1][1]+","+
					substrate.getWeights()[0][2][2][0][2][2]+","+substrate.getWeights()[0][2][2][0][3][3]+","+substrate.getWeights()[0][3][3][0][1][1]+","+
					substrate.getWeights()[0][3][3][0][2][2]+","+substrate.getWeights()[0][3][3][0][3][3]);
		}
		genotype.setPerformanceValue(val);
		return val;
	}

	
	public static double eval(Double[] vector){
		double value = 0;
		for (int i = 0; i < vector.length-2; i++) {
			value +=(1-vector[i])*(1-vector[i])+100*(vector[i+1]-vector[i]*vector[i])*(vector[i+1]-vector[i]*vector[i]);
		}
		return value;
	}
	
	public static double eval(Float x,Float y, double max_value){
		double val = (1-x)*(1-x);
		double tmp = (y-x*x)*(y-x*x);
		return (val+100*tmp)/max_value;
	}
	
	public static double eval(Float x,Float y){
		double val = (1-x)*(1-x);
		double tmp = (y-x*x)*(y-x*x);
		return (val+100*tmp);
	}
	
	
	
}
