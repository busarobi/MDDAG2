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


#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"


using namespace std;
using namespace tbb;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	enum eRolloutType {
		RL_MONTECARLO,
		RL_SZATYMAZ,
		RL_BADSZATYMAZ,
		RL_FULL,
		RL_ONESHOT
	};
	
	enum eRewardType {
		RW_ZEROONE,
		RW_EXPLOSS
	};

	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////

	class PolicyResult {
	public:
		//-------------------------------------------------------------------------------------------
		PolicyResult(void ) :numOfEvaluatedClassifier(0.0), errorRate(0.0), avgReward(0.0) {}
		//-------------------------------------------------------------------------------------------
		PolicyResult( InputData* pData ) :numOfEvaluatedClassifier(0.0), errorRate(0.0), avgReward(0.0) 
		{
			resize( pData);
		}
		//-------------------------------------------------------------------------------------------
		void resize( InputData* pData )
		{
			_numExamples = pData->getNumExamples();
			_numClasses = pData->getNumClasses();			
			
			_margins.resize( _numExamples );
			for ( int i = 0; i < _numExamples; ++i ) _margins[i].resize( _numClasses );
			
			_e01.resize( _numExamples );
			
			_labels.resize( _numExamples );
			for ( int i = 0; i < _numExamples; ++i ) 
			{
				vector<Label> labels = pData->getLabels(i);
				vector<Label>::iterator lIt;
				int l;
				for (l=0, lIt=labels.begin(); lIt != labels.end(); ++lIt, ++l )
				{
					if (lIt->y>0)
					{
						_labels[i]=l;
						break;
					}
				}
			}
		}
		//-------------------------------------------------------------------------------------------
		void setToZero()
		{
			for ( int i = 0; i < _numExamples; ++i ) fill(_margins[i].begin(),_margins[i].end(),0.0);
			fill( _e01.begin(), _e01.end(), 0 );
		}
		//-------------------------------------------------------------------------------------------
		vector<AlphaReal>& getResultVector( const int i ){ return _margins[i]; }		
		//-------------------------------------------------------------------------------------------
		void printResultVector( const int i )
		{ 
			for( int l=0; l<_numClasses; ++l )
				cout << _margins[i][l] << " ";
			cout << endl;
		}						
		//-------------------------------------------------------------------------------------------
		void setClassificationError( const int i, const int res ) { _e01[i]=res; }
		int getClassificationError( const int i ) { return _e01[i]; }
		//-------------------------------------------------------------------------------------------
		void calculateMargins( void ) 
		{  
			_notcorrectlyClassifiedInstances.clear();
			_cumMargin.resize(_numExamples);
			fill(_cumMargin.begin(),_cumMargin.end(),0.0);
			for (int i = 0; i < _numExamples; ++i )
			{
				if (_e01[i]) _notcorrectlyClassifiedInstances.push_back(i);
				for (int l=0; l<_numClasses; ++l )
				{
					_cumMargin[i] += exp(-_margins[i][l]);
				}
				
				// multi-class margin
//				AlphaReal minPos = _margins[i][_labels[i]];
//				AlphaReal maxNeg = -numeric_limits<AlphaReal>::max();
//				for (int l=0; l<_numClasses; ++l )
//				{
//					if ((_labels[i]!=i)&&(_margins[i][l]>maxNeg))
//					{
//						maxNeg = _margins[i][l];
//					}
//				}				
//				_cumMargin[i] = exp(maxNeg-minPos);
			}
			if (_notcorrectlyClassifiedInstances.empty())
			{
				_notcorrectlyClassifiedInstances.resize(_numExamples);
				for (int i = 0; i < _numExamples; ++i ) _notcorrectlyClassifiedInstances[i]=i;
				cout << "WARNING: Training error is zero!!" << endl;
			}
			
			random_shuffle(_cumMargin.begin(),_cumMargin.end());
		}
		//-------------------------------------------------------------------------------------------
		int getRandomIndexOfNotCorrectlyClassifiedInstance( AlphaReal& cumMargin )
		{
			int retval = rand() % _notcorrectlyClassifiedInstances.size();
			retval = _notcorrectlyClassifiedInstances[retval];
			cumMargin = _cumMargin[retval];
			
			return retval;
		}
		//-------------------------------------------------------------------------------------------
		int getRandomIndexOfInstance( AlphaReal& cumMargin )
		{
			int retval = rand() % _numExamples;
			cumMargin = _cumMargin[retval];
			
			return retval;
		}		
		//-------------------------------------------------------------------------------------------
	public:	
		AlphaReal numOfEvaluatedClassifier;
		AlphaReal errorRate;
		AlphaReal avgReward;
		
		int _numExamples;
		int _numClasses;
		
		vector< vector< AlphaReal > > _margins;
		vector< int >				  _e01;
		
		vector< int >				  _notcorrectlyClassifiedInstances;
		vector< AlphaReal >			  _cumMargin;
		vector< int >				  _labels;
	};

	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////

	
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
		_outputTrainingError(false), _epsilon(0.0) {}
		
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
		void rollout( InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy = NULL, PolicyResult* result = NULL );

		void parallelRollout(InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy = NULL, PolicyResult* result = NULL);
		
		AlphaReal getReward( vector<AlphaReal>& margins, InputData* pData, int index );
				
		AlphaReal getErrorRate(InputData* pData, const char* fname, PolicyResult* policyResult );
		AlphaReal parallelGetErrorRate(InputData* pData, const char* fname, PolicyResult* policyResult );
		
		inline virtual void getStateVector( vector<FeatureReal>& state, int iter, vector<AlphaReal>& margins );
    protected:
		inline int normalizeWeights( vector<AlphaReal>& weights );
		inline virtual AlphaReal getNormalizedScores( vector<AlphaReal>& scores, vector<AlphaReal>& normalizedScores, int iter );
		
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
		AlphaReal _epsilon;
		
		
		friend class CalculateErrorRate;
		friend class Rollout;
	};		
	// ------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////				
	class CalculateErrorRate {
	public:	
		CalculateErrorRate(MDDAGLearner* md, InputData* pD,PolicyResult* pR, InputData** sDA,vector<AlphaReal>* rws, vector< vector< int > >* usedCl ) 
		{
			pData=pD;
			policyResult=pR;
			stateDataArray=sDA;			
			mddag = md;
			
			const int numExamples = pData->getNumExamples();		
			classNum = pData->getNumClasses();
			
			
		
			usedClassifier = usedCl;
			
			rewards = rws;
			rewards->resize(numExamples);
			usedClassifier->resize( numExamples );
		}
		
		void operator()( const blocked_range<int>& range ) const {
			for( int i=range.begin(); i!=range.end(); ++i ){	
				InputData* stateData = stateDataArray[i];
				Example& e = stateData->getExampleReference(0);
				vector<FeatureReal>& state = e.getValues();			
				state.resize(classNum);
				
				usedClassifier->at(i).resize(0);
				vector<AlphaReal>& results = policyResult->getResultVector(i);
				
				fill(results.begin(),results.end(),0.0);
				for(int t=0; t< mddag->_foundHypotheses.size(); ++t)
				{
					mddag->getStateVector( state, t, results );								
					
					
					int action = mddag->_policy->getNextAction(stateData);				
					
					if (action==0)
					{
						usedClassifier->at(i).push_back(t);						
						
						AlphaReal currentAlpha = mddag->_foundHypotheses[t]->getAlpha();
						for( int l=0; l<classNum; ++l )
						{
							results[l] += currentAlpha * mddag->_foundHypotheses[t]->classify(pData,i,l);
						}
					} else if (action == 2 )
						break; //quit
					
					
				}
				AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
				int forecastlabel = -1;
				
				for(int l=0; l<classNum; ++l )
				{
					if (results[l]>maxMargin)
					{
						maxMargin=results[l];
						forecastlabel=l;
					}										
				}						
				
				vector<Label> labels = pData->getLabels(i);
				
				int clRes=1;
				
				if (pData->hasLabel(i,forecastlabel) )	
				{
					if(labels[forecastlabel].y<0) 
					{
						clRes=0;						
					}
				} else if (usedClassifier->at(i).size() == 0) 
				{
					clRes=0;					
				}
				else {
					clRes=0;
				}
												
				AlphaReal reward = mddag->getReward(results, pData, i );				
				rewards->at(i) = reward - usedClassifier->at(i).size() * mddag->_beta;								

				// set result
				policyResult->setClassificationError( i, 1-clRes );
			}
		}
		
		// variables
		InputData** stateDataArray;
		PolicyResult* policyResult;
		InputData* pData;
		
		vector<AlphaReal>* rewards;
		
		vector< vector< int > >* usedClassifier;
		int classNum;
		
		MDDAGLearner* mddag;
		
	};
	
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////				

	class Rollout {
	public:	
		Rollout(MDDAGLearner* md, InputData* pD, int rolloutSize, GenericClassificationBasedPolicy* policy ) 
		{
			_pData=pD;
			_mddag = md;
			_policy = policy;
			
			_numExamples = _pData->getNumExamples();		
			_numClasses = _pData->getNumClasses();
			_rolloutSize = rolloutSize;												
		}
		
		void operator()( const blocked_range<int>& range ) const {
			for( int i=range.begin(); i!=range.end(); ++i )
			{	
				InputData* stateData = _stateDataArray[i];
				Example& e = stateData->getExampleReference(0);
				vector<FeatureReal>& state = e.getValues();			
				state.resize(_numClasses);
				
				vector<AlphaReal> estimatedRewardsForActions(_mddag->_actionNumber);
				
				vector< AlphaReal > margins(_numClasses); 
				vector< AlphaReal > storedMargins(_numClasses); 
				
				vector<int> path(0);
				int usedClassifier = 0;
				
				int randWeakLearnerIndex = _weakLearnerIndices->at(i);
				int randIndex = _indices->at(i);
				int action;								
				
				
				for( int t = 0; t < randWeakLearnerIndex; ++t )
				{
					//cout << randWeakLearnerIndex << "\t" << randIndex << "\t" << t << endl; 
					// no quit action
					if (_policy==NULL)
						action = rand() % 2;
					else {
						vector<AlphaReal> distribution(_mddag->_actionNumber);
						_mddag->getStateVector( state, t, margins );
							
						_policy->getExplorationDistribution(stateData, distribution);
							
						if ( nor_utils::is_zero( distribution[0]-distribution[1]))
							action = rand() % 2;
						else 
							(distribution[0]>distribution[1]) ? action=0 : action=1;				
					}
					
					path.push_back( action );
					
					if (action==0) //classify
					{
						for( int l=0; l < _numClasses; ++l)
						{
							margins[l] = margins[l] + _mddag->_foundHypotheses[t]->getAlpha() * _mddag->_foundHypotheses[t]->classify( _pData, randIndex, l );
						}
						usedClassifier++;
					}
					else if (action==1) //skip
					{
					}
					else if (action==2) //quit
					{
						break;
					}
				}	
				
				int currentpathsize = path.size();
				int currentNumberOfUsedClassifier = usedClassifier;
				
				copy( margins.begin(), margins.end(), storedMargins.begin() );
				
				for( int a=0; a<_mddag->_actionNumber; ++a )
				{
					path.resize(currentpathsize);
					usedClassifier = currentNumberOfUsedClassifier;
					copy( storedMargins.begin(), storedMargins.end(), margins.begin() );
					
					for( int t = randWeakLearnerIndex; t < _mddag->_shypIter; ++t )
					{
						if (t == randWeakLearnerIndex)
						{
							action=a;
						} 
						else 
						{															
							if (_policy==NULL)
								action = rand() % _mddag->_actionNumber;
							else {			
								_mddag->getStateVector( state, t, margins );							
								action = _policy->getExplorationNextAction( stateData );							
							}
						}
						
						
						
						path.push_back( action );
						
						if (action==0) //classify
						{
							for( int l=0; l < _numClasses; ++l)
							{
								margins[l] = margins[l] + _mddag->_foundHypotheses[t]->getAlpha() * _mddag->_foundHypotheses[t]->classify( _pData, randIndex, l );
							}
							usedClassifier++;
						}
						else if (action==1) //skip
						{
						}
						else if (action==2) //quit
						{
							break;
						}
					}
					
					AlphaReal finalReward = _mddag->getReward(margins, _pData, randIndex );
					estimatedRewardsForActions[a] = finalReward - usedClassifier * _mddag->_beta;
				}
				
				_mddag->getStateVector( state, randWeakLearnerIndex, storedMargins );
				
				_states->at(i).resize(state.size());
				copy(state.begin(),state.end(),_states->at(i).begin() );
				
				
				_outputFlag->at(i) = _mddag->normalizeWeights( estimatedRewardsForActions );
				_weights->at(i).resize(estimatedRewardsForActions.size());
				copy( estimatedRewardsForActions.begin(), estimatedRewardsForActions.end(), _weights->at(i).begin() );
			}
		}

		MDDAGLearner* _mddag;
		InputData* _pData;				
		int _rolloutSize;
		int _numClasses;		
		int _numExamples;		
		
		InputData** _stateDataArray;
		
		// for output
		vector< vector< AlphaReal > >* _states;
		vector< vector< AlphaReal > >* _weights;	
		vector<int>* _outputFlag;
		
		vector< int >* _indices;
		vector< int >* _weakLearnerIndices;
		
		GenericClassificationBasedPolicy* _policy;
	};
	
	
	
} // end of namespace MultiBoost

#endif // __MDDAG_H

