/*
 *
 *    MultiBoost - Multi-purpose boosting package
 *
 *    Copyright (C)        AppStat group
 *                         Laboratoire de l'Accelerateur Lineaire
 *                         Universite Paris-Sud, 11, CNRS
 *
 *    This file is part of the MultiBoost library
 *
 *    This library is free software; you can redistribute it 
 *    and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
 *
 *    Contact: multiboost@googlegroups.com 
 * 
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
 *
 */


#ifndef __MDDAG_LEARNER_H
#define __MDDAG_LEARNER_H

#include "StrongLearners/GenericStrongLearner.h"
#include "Utils/Args.h"
#include "Defaults.h"
#include "IO/InputData.h"
#include "WeakLearners/BaseLearner.h"
#include "IO/OutputInfo.h"
#include "IO/Serialization.h"
#include "Classifiers/ExampleResults.h"
#include "StrongLearners/AdaBoostMHLearner.h"
#include "Classifiers/AdaBoostMHClassifier.h"
#include "Utils/MDPutils.h"


using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	enum eRolloutType {
		RL_MONTECARLO,
		RL_SZATYMAZ,
		RL_FULL
	};
	
	enum eRewardType {
		RW_ZEROONE,
		RW_EXPLOSS
	};
	
	class PolicyResult {
	public:
		PolicyResult() :numOfEvaluatedClassifier(0.0), errorRate(0.0), avgReward(0.0) {}
		
		AlphaReal numOfEvaluatedClassifier;
		AlphaReal errorRate;
		AlphaReal avgReward;
	};
	

	
	class MDDAGLearner : public GenericStrongLearner
    {
    public:
        
        /**
         * The constructor. It initializes the variables and sets them using the
         * information provided by the arguments passed. They are parsed
         * using the helpers provided by class Args.
         * \date 13/11/2005
         */
        MDDAGLearner()
        : _numIterations(0), _verbose(1), _withConstantLearner(true), _rollouts(10),
        _resumeShypFileName(""), _outputInfoFile(""), _trainingIter(1000), _inshypFileName(""),
		_rolloutType( RL_MONTECARLO ), _actionNumber(2), _rewardtype(RW_ZEROONE), _beta(0.1), _policy(NULL), _outDir(""),
		_outputTrainingError(false) {}
		
        /**
         * Start the learning process.
         * \param args The arguments provided by the command line with all
         * the options for training.
         * \see OutputInfo
         * \date 10/11/2005
         */
        virtual void run(const nor_utils::Args& args);
		
		
        /**
         * Performs the classification using the AdaBoostMHClassifier.
         * \param args The arguments provided by the command line with all
         * the options for classification.
         */
        virtual void classify(const nor_utils::Args& args);
        
        /**
         * Print to stdout (or to file) a confusion matrix.
         * \param args The arguments provided by the command line.
         * \date 20/3/2006
         */
        virtual void doConfusionMatrix(const nor_utils::Args& args);
        
        /**
         * Output the outcome of the strong learner for each class.
         * Strictly speaking these are (currently) not posteriors,
         * as the sum of these values is not one.
         * \param args The arguments provided by the command line.
         */
        virtual void doPosteriors(const nor_utils::Args& args);
        
                
		/**
		 * Compute the results using the weak hypotheses.
		 * This method is the one that effectively computes \f${\bf g}(x)\f$.
		 * \param pData A pointer to the data to be classified.
		 * \param weakHypotheses The list of weak hypotheses.
		 * \param results The vector where the results will be stored.
		 * \see ExampleResults
		 * \date 16/11/2005
		 */
		virtual void computeResults(InputData* pData, vector<BaseLearner*>& weakHypotheses, 
									vector< ExampleResults* >& results );
        
		virtual void getClassError( InputData* pData, const vector<ExampleResults*>& results, AlphaReal& classError);		
		void rollout( InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy = NULL );
		
		AlphaReal getReward( vector<AlphaReal>& margins, InputData* pData, int index );
				
		AlphaReal getErrorRate(InputData* pData, const char* fname, PolicyResult& policyResult );
		
		inline virtual void getStateVector( vector<FeatureReal>& state, int iter, vector<AlphaReal>& margins );
    protected:
		inline int normalizeWeights( vector<AlphaReal>& weights );
		inline AlphaReal getNormalizedScores( vector<AlphaReal>& scores, vector<AlphaReal>& normalizedScores, int iter );
		
        AlphaReal genHeader( ofstream& out, int fnum );
        /**
         * Get the needed parameters (for the strong learner) from the argumens.
         * \param The arguments provided by the command line.
         */
        void getArgs(const nor_utils::Args& args);
        
        
		InputData* getRolloutData( const nor_utils::Args& args, const string fname );
        
        /**
         * Resume the training using the features in _resumeShypFileName if the
         * option -resume has been specified.
         * \date 21/12/2005
         */
        int resumeProcess(const nor_utils::Args& args, InputData* pTestData);
        
        vector<BaseLearner*>  _foundHypotheses; //!< The list of the hypotheses found.
        vector<AlphaReal>	  _sumAlphas;
		
        string  _baseLearnerName; //!< The name of the basic learner used by AdaBoost. 
		string _inBaseLearnerName;
        string  _shypFileName; //!< File name of the strong hypothesis.
		string  _inshypFileName; //!< File name of the strong hypothesis.		
        
        string  _trainFileName;
        string  _testFileName;
        
        int     _maxTime; //!< Time limit for the whole processing. Default: no time limit (-1).
        AlphaReal  _theta; //!< the value of theta. Default = 0.
        
		int     _verbose;
		
		
	protected:
		string  _resumeShypFileName;
		int		_numIterations;
		int		_sampleNum;
		string  _outputInfoFile;
		bool	_withConstantLearner;
		
        /**
         * A temporary variable for h(x)*y. Helps saving time during re-weighting.
         */
        vector< vector<AlphaReal> > _hy;		
		
		bool _fastResumeProcess;
		int  _trainingIter;
		int  _shypIter;
		int  _rollouts;
		ofstream _outStream;
		
		eRolloutType _rolloutType;
		int _actionNumber;
		eRewardType _rewardtype;
		AlphaReal _beta;
		
		GenericClassificationBasedPolicy* _policy;
		string _outDir;
		bool _outputTrainingError;
	};		
	// ------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------
	
} // end of namespace MultiBoost

#endif // __MDDAG_H

