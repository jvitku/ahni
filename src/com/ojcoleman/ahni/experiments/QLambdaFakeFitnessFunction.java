/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ojcoleman.ahni.experiments;

import com.anji.integration.Activator;
import com.ojcoleman.ahni.evaluation.HyperNEATFitnessFunction;
import com.ojcoleman.ahni.experiments.objectrecognition.ObjectRecognitionFitnessFunction2;
import com.ojcoleman.ahni.hyperneat.Properties;
import com.ojcoleman.ahni.nn.GridNet;
import com.ojcoleman.ahni.transcriber.HyperNEATTranscriber;
import java.util.Arrays;
import org.apache.log4j.Logger;
import org.jgapcustomised.Chromosome;

/**
 *
 * @author Admin
 */
public class QLambdaFakeFitnessFunction extends HyperNEATFitnessFunction {
    private static Logger logger = Logger.getLogger(ObjectRecognitionFitnessFunction2.class);

	private double[][][] stimuli;
	private int[][] targetCoords;
	private int maxFitnessValue;

	private final static int numTrials = 100;
	private final static int numSmallSquares = 1;
	private int smallSquareSize = 1;
	private int largeSquareSize = smallSquareSize * 3;

	/**
	 * See <a href=" {@docRoot} /params.htm" target="anji_params">Parameter Details </a> for specific property settings.
	 * 
	 * @param props configuration parameters
	 */
	public void init(Properties props) {
		super.init(props);
		setMaxFitnessValue();
	}

	private void setMaxFitnessValue() {										// * 100000
	}

	public void initialiseEvaluation() {
	}

	/**
	 * Evaluate given individual by presenting the stimuli to the network in a random order to ensure the underlying
	 * network is not memorising the sequence of inputs. Calculation of the fitness based on error is delegated to the
	 * subclass. This method adjusts fitness for network size, based on configuration.
	 */
	protected double evaluate(Chromosome genotype, Activator activator, int threadIndex) {
            GridNet grid = (GridNet)activator;
            Double[] other = new Double[20];
            other[0] = grid.getWeights()[0][0][0][0][0][0];
            other[1] = grid.getWeights()[0][0][1][0][0][0];
            other[2] = grid.getWeights()[0][0][0][0][0][1];
            other[3] = grid.getWeights()[0][0][0][0][0][2];
            other[4] = grid.getWeights()[0][0][0][0][0][3];
            other[5] = grid.getWeights()[0][0][1][0][0][1];
            other[6] = grid.getWeights()[0][0][1][0][0][2];
            other[7] = grid.getWeights()[0][0][1][0][0][3];
            other[8] = grid.getWeights()[0][0][2][0][0][0];
            other[9] = grid.getWeights()[0][0][3][0][0][0];
            other[10] = grid.getWeights()[0][0][4][0][0][0];
            other[11] = grid.getWeights()[0][0][2][0][0][1];
            other[12] = grid.getWeights()[0][0][3][0][0][1];
            other[13] = grid.getWeights()[0][0][4][0][0][1];
            other[14] = grid.getWeights()[0][0][2][0][0][2];
            other[15] = grid.getWeights()[0][0][3][0][0][2];
            other[16] = grid.getWeights()[0][0][4][0][0][2];
            other[17] = grid.getWeights()[0][0][2][0][0][3];
            other[18] = grid.getWeights()[0][0][3][0][0][3];
            other[19] = grid.getWeights()[0][0][4][0][0][3];
            double val = eval(other);
            genotype.setPerformanceValue(val);
            if(val > 0.93){
                System.out.println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
                System.out.println(Arrays.toString(other));
                System.out.println("VZOR:");
                System.out.println(Arrays.toString(vzor));
                System.out.println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
            }
            return val;
	}
        
        private static Float[] vzor={0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
                                    0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f};
        
        private double eval(Double[] other){
            double dist = 0;
            for (int i = 0; i < other.length; i++) {
                dist += Math.abs(other[i]-vzor[i]);
            }
            return Math.abs((dist/20)-1);
        }

}
