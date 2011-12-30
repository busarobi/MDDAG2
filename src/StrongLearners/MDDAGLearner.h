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

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	enum eRolloutType {
		RL_MONTECARLO,
		RL_SZATYMAZ
	};
	
	enum eRewardType {
		RW_ZEROONE,
		RW_EXPLOSS
	};
	
	struct PolicyResult {
		AlphaReal numOfEvaluatedClassifier;
		AlphaReal errorRate;
		AlphaReal avgReward;
	};
	
	class AdaBoostPolicy;
	
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
		_rolloutType( RL_MONTECARLO ), _actionNumber(3), _rewardtype(RW_ZEROONE), _beta(0.1), _policy(NULL), _outDir("") {}
		
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
         * Updates the weights of the examples.
         * The re-weighting of \f$w\f$ (the weight vector over all the examples and classes)
         * is done using the following formula
         * \f[
         *  w^{(t+1)}_{i, \ell}=
         *        \frac{ w^{(t)}_{i, \ell} \exp \left( -\alpha^{(t)} 
         *        h_\ell^{(t)}(x_i) y_{i, \ell} \right) }{ Z^{(t)} }
         * \f]
         * where \a Z is a normalization factor, and it is defined as
         * \f[
         *  Z^{(t)} = 
         *     \sum_{j=1}^n \sum_{\ell=1}^k w^{(t)}_{j, \ell} \exp \left( -\alpha^{(t)} 
         *        h_\ell^{(t)}(x_j) y_{j, \ell} \right) 
         * \f]
         * where \f$n\f$ is the number of examples, \f$k\f$ the number of classes,
         * \f$\alpha\f$ is the confidence in the weak classifier,
         * \f$h_\ell(x_i)\f$ is the classification of example \f$x_i\f$ for class \f$\ell\f$ 
         * with the classifier found at the current iteration (see BaseLearner::classify()), 
         * and \f$y_i\f$ is the binary label of that 
         * example, defined in InputData::getBinaryClass().
         * \param pTrainingData The pointer to the training data.
         * \param pWeakHypothesis The current weak hypothesis.
         * \return The value of the edge. It will be used to see if the algorithm can continue
         * with learning.
         * \date 16/11/2005
         */
        AlphaReal updateWeights(InputData* pTrainingData, BaseLearner* pWeakHypothesis);
        
        /**
         * Updates the weights of the examples. If the slowresumeprocess is on, we do not calculate the 
         * weights in every iteration, but we calculate the re-weighting based on the exponential margins. 
         * Let's assume that it is given a strong learner \f$\f^{(t)}(\bx)\$f after t iteration.
         * Then the weights of the \f$t+1\f$ iteration can be written as
         * \f[
         * w^{(t+1)}_{i, \ell}=  
         *        \frac{ \exp \left( -f\ell^{(t)}(x_i) y_{i, \ell}\right) }
         *         { \sum_{i^{\prime}=1}^{N} \exp \left( -f\ell^{(t)}(x_{i^{\prime}}) y_{{i^{\prime}}, \ell}\right) }
         * \f]
         * Using this formula we avoid the iterationwise normalisation. But in this way we are not able to 
         * calulculate the the error rates in every iteration.
         */
        AlphaReal updateWeights(OutputInfo* pOutInfo, InputData* pData, vector<BaseLearner*>& pWeakHypothesis); //for fast resume process, because we calculate the weights of the samples based on the margin
        
        /**
         * Print output information if option --outputinfo is specified.
         * Called from run and resumeProcess
         * \see resumeProcess
         * \see run
         * \date 21/04/2007
         */
        void printOutputInfo(OutputInfo* pOutInfo, int t, InputData* pTrainingData, 
                             InputData* pTestData, BaseLearner* pWeakHypothesis);
        
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
		void rollout( InputData* pData, const string fname, AdaBoostPolicy* policy = NULL );
		
		AlphaReal getReward( vector<AlphaReal>& margins, InputData* pData, int index );
		
		
		AlphaReal getErrorRate(InputData* pData, const char* fname, PolicyResult& policyResult );
		
		virtual void getStateVector( vector<FeatureReal>& state, int iter, vector<AlphaReal>& margins );
    protected:
		void normalizeWeights( vector<AlphaReal>& weights );
		
		
        AlphaReal genHeader( ofstream& out, int fnum );
        /**
         * Get the needed parameters (for the strong learner) from the argumens.
         * \param The arguments provided by the command line.
         */
        void getArgs(const nor_utils::Args& args);
        
        
		InputData* getRolloutData( const nor_utils::Args& args, const string fname  );
        /**
         * Resume the weak learner list.
         * \return The current iteration number. 0 if not -resume option has been called
         * \param pTrainingData The pointer to the training data, needed for classMap, enumMaps.
         * \date 21/12/2005
         * \see resumeProcess
         * \remark resumeProcess must be called too!
         */
        int resumeWeakLearners(InputData* pTrainingData);
        
        /**
         * Resume the training using the features in _resumeShypFileName if the
         * option -resume has been specified.
         * \date 21/12/2005
         */
        void resumeProcess(Serialization& ss, InputData* pTrainingData, InputData* pTestData, 
                           OutputInfo* pOutInfo);
        
        vector<BaseLearner*>  _foundHypotheses; //!< The list of the hypotheses found.
        
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
		
		AdaBoostPolicy* _policy;
		string _outDir;
	};		
	// ------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------
	class AdaBoostPolicy
	{		
	public:
		//------------------------------------------------------------------------------------------
		AlphaReal trainpolicy( const nor_utils::Args& args, InputData* pTrainingData, const string baseLearnerName, const int numIterations )
		{
			AdaBoostMHLearner* sHypothesis = new AdaBoostMHLearner();
			sHypothesis->run(args, pTrainingData, baseLearnerName, numIterations, _weakhyp );
			delete sHypothesis;		
			_classNum = pTrainingData->getNumClasses();
			
			const int numExamples = pTrainingData->getNumExamples();
			
			vector<AlphaReal> results(_classNum);
			int numErrors = 0;
			for(int i=0; i<numExamples; ++i )				
			{
				fill(results.begin(),results.end(),0.0);
				for(int t=0; t<_weakhyp.size(); ++t)
				{
					for( int l=0; l<_classNum; ++l )
					{
						results[l] += _weakhyp[t]->getAlpha() * _weakhyp[t]->classify(pTrainingData,i,l);
					}
				}
				AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
				int forecastlabel = -1;
				
				for(int l=0; l<_classNum; ++l )
				{
					if (results[l]>maxMargin)
					{
						maxMargin=results[l];
						forecastlabel=l;
					}										
				}						
				
				vector<Label> labels = pTrainingData->getLabels(i);
				
				if (pTrainingData->hasLabel(i,forecastlabel) )	
				{
					if(labels[forecastlabel].y<0) numErrors++;
				} else numErrors++;
			}
			
			AlphaReal error = (AlphaReal)numErrors/(AlphaReal) numExamples;
			return error;			
		}
		//------------------------------------------------------------------------------------------
		virtual int getNextAction( InputData* state )
		{
			vector<AlphaReal> forecast(_classNum);
			fill( forecast.begin(), forecast.end(), 0.0 );
			
			for( int t = 0; t < _weakhyp.size(); ++t )
			{
				for( int l = 0; l<_classNum; ++l )
				{
					forecast[l] += _weakhyp[t]->getAlpha() * _weakhyp[t]->classify(state,0,l);
				}
			}

			AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
			int forecastlabel = -1;
			
			for(int l=0; l<_classNum; ++l )
			{
				if (forecast[l]>maxMargin)
				{
					maxMargin=forecast[l];
					forecastlabel=l;
				}										
			}						
			
			return forecastlabel;
		}
		//------------------------------------------------------------------------------------------
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution )
		{
			distribution.resize(_classNum);
			fill( distribution.begin(), distribution.end(), 0.0 );
			
			for( int t = 0; t < _weakhyp.size(); ++t )
			{
				for( int l = 0; l<_classNum; ++l )
				{
					distribution[l] += _weakhyp[t]->getAlpha() * _weakhyp[t]->classify(state,0,l);
				}
			}
		}
		//------------------------------------------------------------------------------------------
	protected:
		vector<BaseLearner*> _weakhyp;
		int _classNum;
	};
	
} // end of namespace MultiBoost

#endif // __MDDAG_H
