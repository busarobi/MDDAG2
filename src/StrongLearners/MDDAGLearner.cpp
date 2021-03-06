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


#include "MDDAGLearner.h"
#include "Classifiers/ExampleResults.h"
#include "WeakLearners/BaseLearner.h"
#include "WeakLearners/HaarSingleStumpLearner.h"
#include "WeakLearners/SingleStumpLearner.h"

#include "Classifiers/MDDAGClassifier.h"


#define _ADD_SUMOFSCORES_TO_STATESPACE_

namespace MultiBoost {
	// -----------------------------------------------------------------------------------
	
	void MDDAGLearner::getArgs(const nor_utils::Args& args)
	{
		if ( args.hasArgument("verbose") )
			args.getValue("verbose", 0, _verbose);
		
		// The file with the step-by-step information
		if ( args.hasArgument("outputinfo") )
			args.getValue("outputinfo", 0, _outputInfoFile);
		else 
			_outputInfoFile = string(OUTPUT_NAME);
		
		///////////////////////////////////////////////////
		// get the output strong hypothesis file name, if given
		if ( args.hasArgument("shypname") )
			args.getValue("shypname", 0, _shypFileName);
		else
			_shypFileName = string(SHYP_NAME);
		
		_shypFileName = nor_utils::addAndCheckExtension(_shypFileName, SHYP_EXTENSION);
		
		
		///////////////////////////////////////////////////
		// Set time limit
		if ( args.hasArgument("timelimit") )
		{
			args.getValue("timelimit", 0, _maxTime);   
			if (_verbose > 1)    
				cout << "--> Overall Time Limit: " << _maxTime << " minutes" << endl;
		}
		
		// Set the value of theta
		if ( args.hasArgument("edgeoffset") )
			args.getValue("edgeoffset", 0, _theta);  
		
		// Set the filename of the strong hypothesis file in the case resume is
		// called
		if ( args.hasArgument("resume") )
			args.getValue("resume", 0, _resumeShypFileName);
		
		// get the name of the learner
		_baseLearnerName = defaultLearner;
		if ( args.hasArgument("learnertype") )
			args.getValue("learnertype", 0, _baseLearnerName);
		
		// -train <dataFile> <nInterations>
		if ( args.hasArgument("train") )
		{
			args.getValue("train", 0, _trainFileName);
			args.getValue("train", 1, _numIterations);			
		}
		// -traintest <trainingDataFile> <testDataFile> <nInterations>
		else if ( args.hasArgument("traintestmddag") ) 
		{
			args.getValue("traintestmddag", 0, _trainFileName);
			args.getValue("traintestmddag", 1, _testFileName);
			args.getValue("traintestmddag", 2, _inshypFileName);
			args.getValue("traintestmddag", 3, _numIterations);
			args.getValue("traintestmddag", 4, _shypIter);
		}
		
		// --constant: check constant learner in each iteration
		if ( args.hasArgument("constant") )
			_withConstantLearner = true;
		
		// Set the MDP training iter
		if ( args.hasArgument("policytrainingiter") )
			args.getValue("policytrainingiter", 0, _trainingIter);  		
		
		if ( args.hasArgument("rollouts") )
			args.getValue("rollouts", 0, _rollouts);  						

		if ( args.hasArgument("beta") )
			args.getValue("beta", 0, _beta);  								
		
		if ( args.hasArgument("epsilon") )
			args.getValue("epsilon", 0, _epsilon);  						
		
		
		if ( args.hasArgument( "outtrainingerror" ) )
			_outputTrainingError = true;

		
		string rollouttype = "";
		if ( args.hasArgument( "rollouttype" ) )
			args.getValue("rollouttype", 0, rollouttype );   
		
		if ( rollouttype.compare( "montecarlo" ) == 0 )
			_rolloutType = RL_MONTECARLO;
		else if ( rollouttype.compare( "szatymaz" ) == 0 )
			_rolloutType = RL_SZATYMAZ;
		else if ( rollouttype.compare( "badszatymaz" ) == 0 )
			_rolloutType = RL_BADSZATYMAZ;
		else if ( rollouttype.compare( "full" ) == 0 )
			_rolloutType = RL_FULL;
		else if ( rollouttype.compare( "oneshot" ) == 0 )
			_rolloutType = RL_ONESHOT;
		else {
			//cerr << "Unknown update rule in ProductLearnerUCT (set to default [logedge]" << endl;
			_rolloutType = RL_SZATYMAZ;
		}
		
		if (_rolloutType==RL_BADSZATYMAZ)
		{
			cout << "WARNING: The training error has to be caclualted!" << endl;
			cout << "outtrainingerror flag is on!" << endl;
			_outputTrainingError = true;
		}
		
		
		if ( args.hasArgument( "outdir" ) )
			args.getValue("outdir", 0, _outDir );   
				
		string succesrewardtype = "";
		if ( args.hasArgument( "succrewardtype" ) )
			args.getValue("succrewardtype", 0, succesrewardtype );
		else {	
			cerr << "No reward type is given, set to zeroone" << endl;
			succesrewardtype = "e01";
		}
		
		
		if ( succesrewardtype.compare( "exp" ) == 0 )
			_rewardtype = RW_EXPLOSS;
		else if ( succesrewardtype.compare( "e01" ) == 0 )
			_rewardtype = RW_ZEROONE;
		else if ( succesrewardtype.compare( "mcexp" ) == 0 )
			_rewardtype = RW_MCEXPLOSS;
		else {
			cerr << "Unknown success reward type" << endl;
			exit(-1);
		}
		
		// Set the filename of the policy file in the case resume is
		// called
		if ( args.hasArgument("resume") )
			args.getValue("resume", 0, _resumeShypFileName);
		
	}
	
	// -----------------------------------------------------------------------------------
	
	void MDDAGLearner::run(const nor_utils::Args& args)
	{
		// load the arguments
		this->getArgs(args);
		
		if ( !_outputInfoFile.empty() ) 
		{
			
			_outStream.open((_outDir+_outputInfoFile).c_str());
			
			// is it really open?
			if ( !_outStream.is_open() )
			{
				cerr << "ERROR: cannot open the output steam (<" 
				<< _outputInfoFile << ">) for the step-by-step info!" << endl;
				exit(1);
			}			
		}
		
		// get the registered weak learner (type from name)
		_inBaseLearnerName = UnSerialization::getWeakLearnerName(_inshypFileName);		
		BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(_inBaseLearnerName);
		
		// initialize learning options; normally it's done in the strong loop
		// also, here we do it for Product learners, so input data can be created
		pWeakHypothesisSource->initLearningOptions(args);
		
		BaseLearner* pConstantWeakHypothesisSource = 
		BaseLearner::RegisteredLearners().getLearner("ConstantLearner");
		
		// get the training input data, and load it
		
		InputData* pTrainingData = pWeakHypothesisSource->createInputData();
		pTrainingData->initOptions(args);
		pTrainingData->load(_trainFileName, IT_TRAIN, _verbose);
		
		// get the testing input data, and load it
		InputData* pTestData = NULL;
		if ( !_testFileName.empty() )
		{
			pTestData = pWeakHypothesisSource->createInputData();
			pTestData->initOptions(args);
			pTestData->load(_testFileName, IT_TEST, _verbose);
		}
		
		// The output information object
		OutputInfo* pOutInfo = NULL;
		
		// The class that loads the weak hypotheses
		UnSerialization us;
		
		// loads them
		us.loadHypotheses(_inshypFileName, _foundHypotheses, pTrainingData);
		_foundHypotheses.resize(_shypIter);
		
		// calculate sum of alphas for normalization
		_sumAlphas.resize(_shypIter+1);
		_sumAlphas[0]=0.0;
		for (int i=0; i<_shypIter; ++i )
			_sumAlphas[i+1] = _sumAlphas[i] + _foundHypotheses[i]->getAlpha();
		
		// where the results go
		vector< ExampleResults* > results;
		
		if (_verbose > 0)
			cout << "Classifying training data..." << endl << flush;
		
		// where the results go
		vector< ExampleResults* > trainresults;
		
		// get training the results
		computeResults( pTrainingData, _foundHypotheses, trainresults );
		
		AlphaReal trainError;
		getClassError( pTrainingData, trainresults, trainError);
//		_outStream << "Training error: " << trainError << endl;
		if (_verbose > 0 )
			cout << "Training error: " << trainError << endl;
		
		// where the results go
		vector< ExampleResults* > testresults;
		
		// get test the results
		computeResults( pTestData, _foundHypotheses, testresults );
		
		AlphaReal testError;
		getClassError( pTestData, testresults, testError);
//		_outStream << "Test error: " << testError << endl;
		if (_verbose > 0 )
			cout << "Test error: " << testError << endl;
		
		if (_verbose > 0)
			cout << "Done." << endl << flush;
		
		
		if (_verbose>0)
			cout << "Iteration number of input model:\t" << _shypIter << endl;
		_outStream << setprecision (4);
//		_outStream << fixed << trainError << "\t" << testError << endl << flush;
		_outStream << "Iter" << "\t" << "R.Err." << "\t" << "Err." << "\t" << "P.Err." << "\t" << "Evalcl" << "\t" << "Rew."; 
		_outStream << "\t" << "Err." << "\t" << "P.Err." << "\t" << "Evalcl" << "\t" << "Rew." << "\t" << flush << endl;
		
		//cout << "Before serialization" << endl;
		// reload the previously found weak learners if -resume is set. 
		// otherwise just return 0
		
		if (_verbose == 1)
			cout << "Learning in progress..." << endl;
		
		///////////////////////////////////////////////////////////////////////
		// Starting the MDDAG main loop
		///////////////////////////////////////////////////////////////////////
		AlphaReal policyError = 0.0;		
		PolicyResult* policyResultTrain = new PolicyResult( pTrainingData);
		PolicyResult* policyResultTest = new PolicyResult( pTestData);
		
		InputData* rolloutTrainingData;
		
		// create policy 		
		int startingIter = resumeProcess(args, pTrainingData );
		
		char outfilename[4096];
		string rolloutDataFile;
		char tmpFileNameChar[4096];
		string tmpFileName;
		//		cout << "********************************* 0. **********************************" << endl;
		//		
		//		sprintf( tmpFileNameChar, "rollout_%d.txt", 0 );
		//		rolloutDataFile = _outDir + tmpFileNameChar;
		//		
		//		
		//		
		//		if (_verbose>0)
		//			cout << "Rollout..." << endl;
		//		rollout( pTrainingData, rolloutDataFile );
		//		rolloutTrainingData = getRolloutData( args, rolloutDataFile );
		//		
		//		//train policy 
		//		policyError = _policy->trainpolicy( rolloutTrainingData, _baseLearnerName, _trainingIter );		
		//		
		//		// save policy 
		//		sprintf( tmpFileNameChar, "shyp_%d.xml", 0 );
		//		tmpFileName = _outDir + tmpFileNameChar;		
		//		_policy->save(tmpFileName,rolloutTrainingData);
		//		
		//		if (_verbose>0)
		//			cout << "Classifying training." << endl;
		//		sprintf( outfilename, "outtrain_%d.txt", 0 );
		//		tmpFileName = _outDir + outfilename;
		//		//getErrorRate(pTrainingData, tmpFileName.c_str(), policyResultTrain );
		//		
		//		if (_verbose>0)
		//			cout << "Classifying test." << endl;
		//		sprintf( outfilename, "outtest_%d.txt", 0 );
		//		tmpFileName = _outDir + outfilename;
		//		getErrorRate(pTestData,tmpFileName.c_str(), policyResultTest);
		//		
		//		_outStream << "0\t" << policyError; 
		//		_outStream << "\t" << trainError << "\t" << policyResultTrain.errorRate << "\t" << policyResultTrain.numOfEvaluatedClassifier << "\t" << policyResultTrain.avgReward;
		//		_outStream << "\t" << testError << "\t" << policyResultTest.errorRate << "\t" << policyResultTest.numOfEvaluatedClassifier << "\t" << policyResultTest.avgReward << "\t";
		//		_outStream << endl << flush;
		//		
		//		cout << "Policy training error:\t" << policyError << endl;
		//		cout << "Error (train/test):\t" << policyResultTrain.errorRate << "\t" << policyResultTest.errorRate << endl;
		//		cout << "Num of evaluated BL (train/test):\t" << policyResultTrain.numOfEvaluatedClassifier << "\t" << policyResultTest.numOfEvaluatedClassifier << endl << flush;
		//		cout << "result filename: " << outfilename << endl;
		//		
		//		delete rolloutTrainingData;
		
		for (int t = startingIter; t < _numIterations; ++t)
		{
			cout << "********************************* " << (t+1) << ". **********************************" << endl;
			
			// increase the complexity of policy/classifier
			//_trainingIter += 10;
			cout << "Policy iteration: " << _trainingIter << endl;
			
			// do rollout
			if (_verbose>0)
				cout << "Rollout..." << endl;
			sprintf( tmpFileNameChar, "rollout_%d.txt", t+1 );
			rolloutDataFile = _outDir + tmpFileNameChar;
			if (t==0)
			{
				// the first rollout is fast, thus we generate a lot of rollout instance
				//rollout( args, pTrainingData, rolloutDataFile, 3 * _rollouts, _policy );
				parallelRollout( args, pTrainingData, rolloutDataFile, _rollouts, _policy );
			}
			else 
			{
				//rollout(args, pTrainingData, rolloutDataFile, _rollouts, _policy, policyResultTrain );
				parallelRollout( args, pTrainingData, rolloutDataFile, _rollouts, _policy, policyResultTrain );
			}
			
			
			rolloutTrainingData = getRolloutData( args, rolloutDataFile );						
			
			// train policy
			policyError = _policy->trainpolicy( rolloutTrainingData, _baseLearnerName, _trainingIter );			
			
			// save policy 
			sprintf( tmpFileNameChar, "shyp_%d.xml", t+1 );
			tmpFileName = _outDir + tmpFileNameChar;		
			_policy->save(tmpFileName,rolloutTrainingData);			
			
			if (_verbose>0)
				cout << "Classifying training." << endl;			
			sprintf( outfilename, "outtrain_%d.txt", t+1 );
			tmpFileName = _outDir + outfilename;
			
			if (_outputTrainingError)
				//getErrorRate(pTrainingData, tmpFileName.c_str(), policyResultTrain);
				parallelGetErrorRate(pTrainingData, tmpFileName.c_str(), policyResultTrain);
			
			if (_verbose>0)
				cout << "Classifying test." << endl;			
			sprintf( outfilename, "outtest_%d.txt", t+1 );
			tmpFileName = _outDir + outfilename;
			//getErrorRate(pTestData, tmpFileName.c_str(), policyResultTest);
			parallelGetErrorRate(pTestData, tmpFileName.c_str(), policyResultTest);
			
			_outStream << (t+1) << "\t" << policyError; 
			_outStream << fixed << "\t" << trainError << "\t" << policyResultTrain->errorRate << "\t" << policyResultTrain->numOfEvaluatedClassifier << "\t" << policyResultTrain->avgReward;
			_outStream << fixed << "\t" << testError << "\t" << policyResultTest->errorRate << "\t" << policyResultTest->numOfEvaluatedClassifier << "\t" << policyResultTest->avgReward << "\t";
			_outStream << endl << flush;
			
			
			cout << "Policy training error:\t" << policyError << endl;
			cout << "Error: (train/test): " << trainError << "\t" << testError << endl << flush;
			cout << "Error (train/test):\t" << policyResultTrain->errorRate << "\t" << policyResultTest->errorRate << endl;
			cout << "Num of evaluated BL (train/test):\t" << policyResultTrain->numOfEvaluatedClassifier << "\t" << policyResultTest->numOfEvaluatedClassifier << endl << flush;
			cout << "result filename: " << outfilename << endl;
			
			delete rolloutTrainingData;
		}  // loop on iterations
		/////////////////////////////////////////////////////////
		
		
		
		// Free the two input data objects
		if (pTrainingData)
			delete pTrainingData;
		if (pTestData)
			delete pTestData;
		
		if (pOutInfo)
			delete pOutInfo;
		
		if (_verbose > 0)
			cout << "Learning completed." << endl;
	}	
	
	// -------------------------------------------------------------------------
	void MDDAGLearner::rollout(const nor_utils::Args& args, InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy, PolicyResult* result )
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
		
		vector< vector<AlphaReal> > margins(_shypIter+1);		
		vector<AlphaReal> path(_shypIter);	
		vector<AlphaReal>::iterator pIt;
		
		vector<int> labelDistribution(_actionNumber,0);
		int rolloutSize=0;
		
		int usedClassifier;
		int randIndex;
		int randWeakLearnerIndex;
		int action;
		int currentpathsize;
		int currentNumberOfUsedClassifier;
		
		vector<int> randomPermutation;
		vector<int> randWeakLearnerOrder;
		
		AlphaReal finalReward;
		AlphaReal reward;
		AlphaReal mddagMargin;
		vector<AlphaReal> estimatedRewardsForActions(_actionNumber);
		
		InputData* data = new InputData();
		Example stateExample("state");
		data->addExample(stateExample);
		
		Example& e = data->getExampleReference(0);
		vector<FeatureReal>& state = e.getValues();
		
		
		ofstream rolloutStream;
		rolloutStream.open( fname.c_str() );
		if (!rolloutStream.is_open())
		{
			cout << "Cannot open rollout file" << endl;
			exit(-1);
		}
		
		
		for( int i=0; i<=_shypIter; ++i )
		{
			margins[i].resize(numClasses);
		}
		
		// gen header
//		if ( _inBaseLearnerName.compare( "HaarSingleStumpLearner" ) ==0 )
//		{
//#ifdef _ADD_SUMOFSCORES_TO_STATESPACE_			
//			genHeader(rolloutStream, numClasses+5);
//#else
//			genHeader(rolloutStream, numClasses+4);
//#endif
//		} else if ( _inBaseLearnerName.compare( "SingleStumpLearner" ) ==0 )
//		{
//			genHeader(rolloutStream, numClasses+3);
//		}						
//		else if ( _inBaseLearnerName.compare( "TreeLearner" ) ==0 ) // with SingleDecisionStump
//		{
//			genHeader(rolloutStream, numClasses+3);
//		}						
		genHeader(rolloutStream, numClasses+1);
		
		
		// full rollout
		int currentExample = 0;
		int currentWeakLearner = -1;
		if (_rolloutType==RL_FULL)
		{
			if (_verbose>1)
				cout << "WARNING: full rollout is used, the rollout set size is " << numExamples * _shypIter << endl;
			rsize = numExamples * _shypIter;			
		}		
		
		if (_rolloutType==RL_ONESHOT)
		{
			if (_verbose>1)
				cout << "WARNING: one-shot rollout is used, the rollout set size is at most " << numExamples << endl;
			rsize = numExamples;			
			randomPermutation.resize( numExamples );
			for (int i = 0; i < numExamples; ++i ) randomPermutation[i]=i;
			random_shuffle( randomPermutation.begin(), randomPermutation.end() );
			randWeakLearnerOrder.resize(_shypIter);
		}		
		
		if (_rolloutType==RL_BADSZATYMAZ)
		{
			if (result) result->calculateMargins();
		}
		
		for( int rlI = 0; rlI < rsize; ++rlI )
		{			
			
			switch (_rolloutType) {
				case RL_MONTECARLO :
					randIndex = rand() % numExamples;								
					
					fill(margins[0].begin(), margins[0].end(), 0.0 );
					path.resize(0);
					usedClassifier = 0;
					
					for( int t = 0; t < _shypIter; ++t )
					{
						if (policy==NULL)
							action = rand() % _actionNumber;
						else {
							/////////////
							//exploration
							/////////////							
							//							float r = (float)rand() / RAND_MAX;
							//							if (r<0.3)
							//							{
							//								action = rand() % _actionNumber;							
							//							}
							//							else {
							//								getStateVector( state, t, margins[t] );							
							//								action = policy->getNextAction( data, t );
							//							}
							/////////////							
							//exploration
							/////////////
							
							getStateVector( state, t, margins[t] );							
							action = policy->getNextAction( data, t );
							
						}
						
						path.push_back( action );
						
						if (action==0) //classify
						{
							for( int l=0; l < numClasses; ++l)
							{
								margins[t+1][l] = margins[t][l] + _foundHypotheses[t]->getAlpha() * _foundHypotheses[t]->classify( pData, randIndex, l );
							}
							usedClassifier++;
						}
						else if (action==1) //skip
						{
							for( int l=0; l < numClasses; ++l)
								margins[t+1][l] = margins[t][l];							
						}
						else if (action==2) //quit
						{
							for( int l=0; l < numClasses; ++l)
								margins[t+1][l] = margins[t][l];
							break;
						}
					}
					
					finalReward = getReward(margins[path.size()], pData, randIndex );
					reward = finalReward - usedClassifier * _beta;
					
					for ( int t = 0; t < path.size(); ++t )
					{
						getStateVector( state, t, margins[t+1] );
						
						for( int j=0; j<state.size(); ++j )
						{
							rolloutStream << state[j] << ",";
						}
						
						rolloutStream << "{ ";
						for( int j=0; j<_actionNumber; ++j)
						{
							if (j==path[t])
								rolloutStream << path[t] << " " << reward << " "; 
							else
								rolloutStream << path[t] << " " << 0.0 << " "; 
						}
						
						rolloutStream << "}" << " # " << rlI;
						rolloutStream << endl;							
					}
					
					break;
				case RL_SZATYMAZ:
				case RL_BADSZATYMAZ:
				case RL_FULL:
					if (_rolloutType==RL_FULL) {
						currentWeakLearner++;
						if (currentWeakLearner>=(_shypIter-1))
						{
							currentExample++;
							currentWeakLearner=0;
						}
						
						if (currentExample>=numExamples) break;
						
						randIndex = currentExample;
						randWeakLearnerIndex = currentWeakLearner;
						mddagMargin = 1.0;
					} else if (_rolloutType==RL_BADSZATYMAZ)
					{
						if (result) 
						{
							//randIndex = result->getRandomIndexOfNotCorrectlyClassifiedInstance(mddagMargin);
							randIndex = result->getRandomIndexOfInstance(mddagMargin);
						}else {
							randIndex = rand() % numExamples; 
							mddagMargin = 1.0;
						}
						randWeakLearnerIndex = rand() % _shypIter;						
					}else {
						randIndex = rand() % numExamples;								
						randWeakLearnerIndex = rand() % _shypIter;								
						mddagMargin = 1.0;
					}					
					
					fill(margins[0].begin(), margins[0].end(), 0.0 );
					path.resize(0);
					usedClassifier = 0;
					
					for( int t = 0; t < randWeakLearnerIndex; ++t )
					{
						// no quit action
						if (policy==NULL)
							action = rand() % 2;
						else {
							// random \psilon exploration
							float r = (float)rand() / RAND_MAX;							
							if (r<_epsilon)
							{
								action = rand() % 2;
							} else {															
								vector<AlphaReal> distribution(_actionNumber);
								getStateVector( state, t, margins[t] );
								//vector<FeatureReal>& values = data->getValues(0);
								//for (int tmpv=0; tmpv < 15; ++tmpv) cout << data->getValue(0,tmpv) << " ";
								//cout << endl;
								
								policy->getExplorationDistribution(data, distribution);
								
								if ( nor_utils::is_zero( distribution[0]-distribution[1]))
									action = rand() % 2;
								else 
									(distribution[0]>distribution[1]) ? action=0 : action=1;
							}
						}
						
						path.push_back( action );
						
						if (action==0) //classify
						{
							for( int l=0; l < numClasses; ++l)
							{
								margins[t+1][l] = margins[t][l] + _foundHypotheses[t]->getAlpha() * _foundHypotheses[t]->classify( pData, randIndex, l );
							}
							usedClassifier++;
						}
						else if (action==1) //skip
						{
							for( int l=0; l < numClasses; ++l)
								margins[t+1][l] = margins[t][l];							
						}
						else if (action==2) //quit
						{
							for( int l=0; l < numClasses; ++l)
								margins[t+1][l] = margins[t][l];
							break;
						}
					}	
					
					currentpathsize = path.size();
					currentNumberOfUsedClassifier = usedClassifier;
					for( int a=0; a<_actionNumber; ++a )
					{
						path.resize(currentpathsize);
						usedClassifier = currentNumberOfUsedClassifier;
						for( int t = randWeakLearnerIndex; t < _shypIter; ++t )
						{
							if (t == randWeakLearnerIndex)
							{
								action=a;
							} 
							else 
							{															
								if (policy==NULL)
									action = rand() % _actionNumber;
								else {			
									getStateVector( state, t, margins[t] );							
									action = policy->getExplorationNextAction( data, t );							
								}
							}
							
							
							
							path.push_back( action );
							
							if (action==0) //classify
							{
								for( int l=0; l < numClasses; ++l)
								{
									margins[t+1][l] = margins[t][l] + _foundHypotheses[t]->getAlpha() * _foundHypotheses[t]->classify( pData, randIndex, l );
								}
								usedClassifier++;
							}
							else if (action==1) //skip
							{
								for( int l=0; l < numClasses; ++l)
									margins[t+1][l] = margins[t][l];							
							}
							else if (action==2) //quit
							{
								for( int l=0; l < numClasses; ++l)
									margins[t+1][l] = margins[t][l];
								break;
							}
						}
						
						finalReward = getReward(margins[path.size()], pData, randIndex );
						estimatedRewardsForActions[a] = finalReward - usedClassifier * _beta;
					}
					
					getStateVector( state, randWeakLearnerIndex, margins[randWeakLearnerIndex+1] );
					
					
					if (normalizeWeights( estimatedRewardsForActions )) 
					{
						
						for( int j=0; j<state.size(); ++j )
						{
							rolloutStream << state[j] << ",";
						}
						
						
						rolloutStream << "{ ";
						for( int a=0; a<_actionNumber; ++a)
						{							
							rolloutStream << a << " " << (mddagMargin * estimatedRewardsForActions[a]) << " "; 
						}
						
						rolloutStream << "}" << " # " << rlI << " " << randIndex << " " << randWeakLearnerIndex ;
						rolloutStream << endl;
						
						for( int a=0; a<_actionNumber; ++a)
						{
							if (estimatedRewardsForActions[a]>0) {
								labelDistribution[a]++;
							}
						}
						
						rolloutSize++;
					}
					break;
				case RL_ONESHOT:
					randIndex = randomPermutation[rlI];					
					
					for (int i = 0; i < _shypIter; ++i ) randWeakLearnerOrder[i]=i;
					random_shuffle( randWeakLearnerOrder.begin(), randWeakLearnerOrder.end() );						
					
					for ( int wInd=0; wInd < _shypIter; ++wInd )
					{
						randWeakLearnerIndex = randWeakLearnerOrder[wInd];								
						
						fill(margins[0].begin(), margins[0].end(), 0.0 );
						path.resize(0);
						usedClassifier = 0;
						
						for( int t = 0; t < randWeakLearnerIndex; ++t )
						{
							// no quit action
							if (policy==NULL)
								action = rand() % 2;
							else {
								// random \psilon exploration
								//float r = (float)rand() / RAND_MAX;
								float r=1.0;
								if (r<0.0)
								{
									action = rand() % 2;
								} else {															
									vector<AlphaReal> distribution(_actionNumber);
									getStateVector( state, t, margins[t] );
									//vector<FeatureReal>& values = data->getValues(0);
									//for (int tmpv=0; tmpv < 15; ++tmpv) cout << data->getValue(0,tmpv) << " ";
									//cout << endl;
									
									policy->getExplorationDistribution(data, distribution);
									
									if ( nor_utils::is_zero( distribution[0]-distribution[1]))
										action = rand() % 2;
									else 
										(distribution[0]>distribution[1]) ? action=0 : action=1;
								}
							}
							
							path.push_back( action );
							
							if (action==0) //classify
							{
								for( int l=0; l < numClasses; ++l)
								{
									margins[t+1][l] = margins[t][l] + _foundHypotheses[t]->getAlpha() * _foundHypotheses[t]->classify( pData, randIndex, l );
								}
								usedClassifier++;
							}
							else if (action==1) //skip
							{
								for( int l=0; l < numClasses; ++l)
									margins[t+1][l] = margins[t][l];							
							}
							else if (action==2) //quit
							{
								for( int l=0; l < numClasses; ++l)
									margins[t+1][l] = margins[t][l];
								break;
							}
						}	
						
						currentpathsize = path.size();
						currentNumberOfUsedClassifier = usedClassifier;
						for( int a=0; a<_actionNumber; ++a )
						{
							path.resize(currentpathsize);
							usedClassifier = currentNumberOfUsedClassifier;
							for( int t = randWeakLearnerIndex; t < _shypIter; ++t )
							{
								if (t == randWeakLearnerIndex)
								{
									action=a;
								} 
								else 
								{															
									if (policy==NULL)
										action = rand() % _actionNumber;
									else {			
										float r = (float)rand() / RAND_MAX;
										if (r<0.0)
										{
											action = rand() % _actionNumber;
										} else {
											getStateVector( state, t, margins[t] );							
											action = policy->getExplorationNextAction( data, t );							
										}
									}
								}
								
								
								
								path.push_back( action );
								
								if (action==0) //classify
								{
									for( int l=0; l < numClasses; ++l)
									{
										margins[t+1][l] = margins[t][l] + _foundHypotheses[t]->getAlpha() * _foundHypotheses[t]->classify( pData, randIndex, l );
									}
									usedClassifier++;
								}
								else if (action==1) //skip
								{
									for( int l=0; l < numClasses; ++l)
										margins[t+1][l] = margins[t][l];							
								}
								else if (action==2) //quit
								{
									for( int l=0; l < numClasses; ++l)
										margins[t+1][l] = margins[t][l];
									break;
								}
							}
							
							finalReward = getReward(margins[path.size()], pData, randIndex );
							estimatedRewardsForActions[a] = finalReward - usedClassifier * _beta;
						}
						
						getStateVector( state, randWeakLearnerIndex, margins[randWeakLearnerIndex+1] );
						
						
						if (normalizeWeights( estimatedRewardsForActions )) 
						{
							
							for( int j=0; j<state.size(); ++j )
							{
								rolloutStream << state[j] << ",";
							}
							
							
							rolloutStream << "{ ";
							for( int a=0; a<_actionNumber; ++a)
							{							
								rolloutStream << a << " " << estimatedRewardsForActions[a] << " "; 
							}
							
							rolloutStream << "}" << " # " << rlI << " " << randIndex << " " << randWeakLearnerIndex ;
							rolloutStream << endl;
							
							for( int a=0; a<_actionNumber; ++a)
							{
								if (estimatedRewardsForActions[a]>0) {
									labelDistribution[a]++;
								}
							}
							
							rolloutSize++;
							break;
						}
					}
					break;
					
				default:
					break;
			}
		}
		
		
		if (_verbose)
		{
			int sumPosLabels=0;
			for( int a=0; a<_actionNumber; ++a)
			{
				sumPosLabels += labelDistribution[a];
			}			
			for( int a=0; a<_actionNumber; ++a)
			{			
				AlphaReal dist = static_cast<AlphaReal>(labelDistribution[a]) / static_cast<AlphaReal>(sumPosLabels);
				cout << "Action (" << a << "):" << dist <<endl;
			}
			cout << "-->Num of rollout instance: " << rolloutSize << endl; 
			cout << "-->Num. of pos. labels:     " << sumPosLabels << endl;
		}
		
		rolloutStream.close();
	}
	
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::parallelRollout(const nor_utils::Args& args, InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy, PolicyResult* result, const int weakLearnerPostion)
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
		
		vector< vector<AlphaReal> > margins(_shypIter+1);		
		
		vector<int> labelDistribution(_actionNumber,0);
		vector<int> randomPermutation;
		vector<int> randWeakLearnerOrder;
				
		if (_rolloutType==RL_FULL)	rsize = numExamples * _shypIter;
			
		for( int i=0; i<=_shypIter; ++i )
		{
			margins[i].resize(numClasses);
		}
						
		// create thread
		Rollout rollout(this,pData,rsize, policy);

		// create state data
		InputData** stateDataArray = new InputData*[rsize];
		for (int i=0; i<rsize; ++i ) 
		{	
			stateDataArray[i] = new InputData();
			Example stateExample("state");
			stateDataArray[i]->addExample(stateExample);
		}
		rollout._stateDataArray=stateDataArray;
		
		// create rollut data
		vector< vector< AlphaReal> >* states = new vector< vector<AlphaReal > >(rsize);
		rollout._states = states;
		
		//create weights
		vector< vector< AlphaReal > >* weights = new vector< vector<AlphaReal > >(rsize);
		rollout._weights = weights;
		
		vector< int >* indices = NULL;
		vector< int >* weakLearnerIndices = NULL;
		vector<AlphaReal> mddagMargin( rsize );
		if (_rolloutType==RL_SZATYMAZ)
		{
			indices = new vector<int>(rsize);
			weakLearnerIndices = new vector<int>(rsize);
			for (int ri = 0; ri < rsize; ++ri )
			{
				indices->at(ri) = rand() % numExamples;								
				
				mddagMargin[ri] = 1.0;				
				if (weakLearnerPostion<0)
				{
					weakLearnerIndices->at(ri) = rand() % _shypIter;						
				} else {
					weakLearnerIndices->at(ri) = weakLearnerPostion;
				}				
			}			
		} else if (_rolloutType==RL_BADSZATYMAZ) 
		{
			if (result) result->calculateMargins();
			
			indices = new vector<int>(rsize);
			weakLearnerIndices = new vector<int>(rsize);
			for (int ri = 0; ri < rsize; ++ri )
			{
				
				if (result) 
				{		
					AlphaReal tmp;
					indices->at(ri) = result->getRandomIndexOfInstance(tmp);
					//indices->at(ri) = result->getRandomIndexOfNotCorrectlyClassifiedInstance(tmp);
					mddagMargin[ri] = tmp;
				}else {
					indices->at(ri) = rand() % numExamples; 
					mddagMargin[ri] = 1.0;
				}
				if (weakLearnerPostion<0)
				{
					weakLearnerIndices->at(ri) = rand() % _shypIter;						
				} else {
					weakLearnerIndices->at(ri) = weakLearnerPostion;
				}											
			}
		} else if (_rolloutType==RL_FULL) 
		{
			if (weakLearnerPostion==-1)
			{
				cout << "Full rollout cannot be used with multiMDDAG" << endl;
				exit(-1);
			}
				
			indices = new vector<int>(numExamples*_shypIter);
			weakLearnerIndices = new vector<int>(numExamples*_shypIter);
			int tmpi = 0;
			for (int ri = 0; ri < numExamples; ++ri )
			{
				for (int si = 0; si < _shypIter; ++si )
				{
					weakLearnerIndices->at(tmpi) = si;						
					indices->at(tmpi) = ri; 
					mddagMargin[tmpi] = 1.0;
					tmpi++;
				}				
			}			
		} else {
			cout << "paralell version is not impelemented" << endl;
			exit(-1);
		}
		
		rollout._indices = indices;
		rollout._weakLearnerIndices = weakLearnerIndices;
		vector<int>* outputFlag = new vector<int>(rsize);
		rollout._outputFlag = outputFlag;
		
		parallel_for( blocked_range<int>( 0, rsize ), rollout );
		
		// output rollout set
		ofstream rolloutStream;
		rolloutStream.open( fname.c_str() );
		if (!rolloutStream.is_open())
		{
			cout << "Cannot open rollout file" << endl;
			exit(-1);
		}
		// gen header
//		if ( _inBaseLearnerName.compare( "HaarSingleStumpLearner" ) ==0 )
//		{
//#ifdef _ADD_SUMOFSCORES_TO_STATESPACE_			
//			genHeader(rolloutStream, numClasses+2);
//#else
//			genHeader(rolloutStream, numClasses+4);
//#endif
//		} if ( _inBaseLearnerName.compare( "SingleStumpLearner" ) ==0 )
//		{
//			genHeader(rolloutStream, numClasses+3);
//		}						
		genHeader(rolloutStream, numClasses+1);
		
		int rolloutSize=0;
		for (int i=0; i<rsize; ++i )
		{
			if (rollout._outputFlag->at(i)) 
			{
				vector<AlphaReal> &state = states->at(i);
				for( int j=0; j<state.size(); ++j )
				{
					rolloutStream << state[j] << ",";
				}
				
				vector<AlphaReal> & estimatedRewardsForActions = weights->at(i);
				rolloutStream << "{ ";
				for( int a=0; a<_actionNumber; ++a)
				{							
					rolloutStream << a << " " << (mddagMargin[i] * estimatedRewardsForActions[a]) << " "; 
				}
				
				rolloutStream << "}" << " # " << i << " " << indices->at(i) << " " << weakLearnerIndices->at(i) ;
				rolloutStream << endl;
				
				for( int a=0; a<_actionNumber; ++a)
				{
					if (estimatedRewardsForActions[a]>0) {
						labelDistribution[a]++;
					}
				}
				
				rolloutSize++;
			}
		}
		if (_verbose)
		{
			int sumPosLabels=0;
			for( int a=0; a<_actionNumber; ++a)
			{
				sumPosLabels += labelDistribution[a];
			}			
			for( int a=0; a<_actionNumber; ++a)
			{			
				AlphaReal dist = static_cast<AlphaReal>(labelDistribution[a]) / static_cast<AlphaReal>(sumPosLabels);
				cout << "Action (" << a << "):" << dist <<endl;
			}
			cout << "-->Num of rollout instance: " << rolloutSize << endl; 
			cout << "-->Num. of pos. labels:     " << sumPosLabels << endl;
		}
		
		
		// delete
		for (int i=0; i<rsize; ++i ) 
		{	
			delete stateDataArray[i];
		}			
		delete stateDataArray;
		
		//vector< vector< AlphaReal > >* _states;
		states->clear();
		delete states;
		//vector< vector< AlphaReal > >* _weights;	
		weights->clear();
		delete weights;
		delete outputFlag;		
		delete indices;
		delete weakLearnerIndices;
		
	}
	
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	int MDDAGLearner::normalizeWeights( vector<AlphaReal>& weights )
	{
		//		int indMax;
		//		AlphaReal maxVal = -numeric_limits<AlphaReal>::max();
		//		
		//		for (int i=0; i<weights.size(); ++i )
		//		{
		//			if (weights[i]>maxVal)
		//			{
		//				indMax=i;
		//				maxVal=weights[i];
		//			}
		//		}
		//		
		//		for (int i=0; i<weights.size(); ++i )
		//		{
		//			if (indMax!=i)
		//			{
		//				weights[i] = weights[i] - maxVal;
		//			}
		//		}			
		
		AlphaReal minValue = numeric_limits<AlphaReal>::max();
		AlphaReal maxValue = -numeric_limits<AlphaReal>::max();
		AlphaReal sumWeight = 0.0;
		
		for (int i=0; i<weights.size(); ++i )
		{
			if (weights[i]<minValue)
			{
				minValue = weights[i];
			}
			if (weights[i]>maxValue)
			{
				maxValue = weights[i];
			}
			
			sumWeight += weights[i];
		}				
		
		int notAllZero = 1;
		
		// for two action these two weight normalization are the same
		//AlphaReal avgWeight = sumWeight / static_cast<AlphaReal>(weights.size()); // L_{2}
		AlphaReal avgWeight = (maxValue + minValue) / 2.0; // L_{infty}
		for (int i=0; i<weights.size(); ++i )
		{
			weights[i] = weights[i] - avgWeight;
			if (nor_utils::is_zero(weights[i]))
				notAllZero = 0;
		}								
		
		return notAllZero;
	}

	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::parallelGetErrorRate(InputData* pData, const char* fname, PolicyResult* policyResult )
	{
		const int numExamples = pData->getNumExamples();		
		const int classNum = pData->getNumClasses();
		
		//vector<AlphaReal> results(classNum);
		policyResult->setToZero();
		
		InputData** stateDataArray = new InputData*[numExamples];
		for (int i=0; i<numExamples; ++i ) 
		{	
			stateDataArray[i] = new InputData();
			Example stateExample("state");
			stateDataArray[i]->addExample(stateExample);
		}
		
		AlphaReal sumReward = 0.0;
		
		int numErrors = 0;
				
		int overAllUsedClassifier = 0;
		vector<vector< int > >* usedClassifier = new vector< vector< int > >(numExamples);		
		vector< AlphaReal >* rewards = new vector<AlphaReal>(numExamples);
		
		CalculateErrorRate callculateError(this, pData,policyResult, stateDataArray, rewards, usedClassifier );
		parallel_for( blocked_range<int>( 0, numExamples ), callculateError );
		
		ofstream out;
		out.open(fname );				
		for(int i=0; i<numExamples; ++i )				
		{
			// output
			out << (1-policyResult->getClassificationError(i)) << "  ";
			
			vector<AlphaReal>& results = policyResult->getResultVector(i);
			
			for( int l=0; l<classNum; ++l)
				out << results[l] << " ";
			
			for( int t=0; t<usedClassifier->at(i).size(); ++t)
				out << usedClassifier->at(i)[t] << " ";
			out << endl << flush;
						
		}			
		out.close();
		
		sumReward=0.0;
		for (int i=0; i<numExamples; ++i ) 
		{
			vector<AlphaReal>& results = policyResult->getResultVector(i);
			sumReward += rewards->at(i);
			numErrors += policyResult->getClassificationError(i);
			overAllUsedClassifier += usedClassifier->at(i).size();
			
			// set result
			vector<Label> labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			for( lIt=labels.begin(); lIt!=labels.end(); ++lIt )
			{
				results[lIt->idx] *= lIt->y;
			}						
		}
		
		
		for (int i=0; i<numExamples; ++i ) 
		{
			delete stateDataArray[i];
			usedClassifier->at(i).clear();
		}
		delete usedClassifier;
		delete stateDataArray;
		
		rewards->clear();
		delete rewards;
								
		policyResult->errorRate = (AlphaReal)numErrors/(AlphaReal) numExamples;
		policyResult->avgReward = sumReward/(AlphaReal) numExamples;
		policyResult->numOfEvaluatedClassifier = (AlphaReal)overAllUsedClassifier/(AlphaReal) numExamples;
		
		return 0.0;		
	}
	
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::getErrorRate(InputData* pData, const char* fname, PolicyResult* policyResult )
	{
		const int numExamples = pData->getNumExamples();		
		const int classNum = pData->getNumClasses();
		
		//vector<AlphaReal> results(classNum);
		policyResult->setToZero();
		
		InputData* stateData = new InputData();
		Example stateExample("state");
		stateData->addExample(stateExample);
		
		Example& e = stateData->getExampleReference(0);
		vector<FeatureReal>& state = e.getValues();
		
		state.resize(classNum);
		AlphaReal sumReward = 0.0;
		
		int numErrors = 0;
		
		ofstream out;
		out.open(fname );		
		
		vector<int> usedClassifier;		
		int overAllUsedClassifier = 0;
		for(int i=0; i<numExamples; ++i )				
		{
			usedClassifier.resize(0);
			vector<AlphaReal>& results = policyResult->getResultVector(i);
			
			fill(results.begin(),results.end(),0.0);
			for(int t=0; t<_foundHypotheses.size(); ++t)
			{
				getStateVector( state, t, results );								
				
				
				int action = _policy->getNextAction(stateData, t);				
				
				if (action==0)
				{
					usedClassifier.push_back(t);
					overAllUsedClassifier++;
					
					AlphaReal currentAlpha = _foundHypotheses[t]->getAlpha();
					for( int l=0; l<classNum; ++l )
					{
						results[l] += currentAlpha * _foundHypotheses[t]->classify(pData,i,l);
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
					numErrors++;
				}
			} else if (usedClassifier.size() == 0) 
			{
				clRes=0;
				numErrors++;
			}
			else {
				clRes=0;
				numErrors++;
			}
						
			// output
			out << clRes << "  ";
			for( int l=0; l<classNum; ++l)
				out << results[l] << " ";
			
			for( int t=0; t<usedClassifier.size(); ++t)
				out << usedClassifier[t] << " ";
			out << endl << flush;
			
			AlphaReal reward = getReward(results, pData, i );
			reward = reward - usedClassifier.size() * _beta;
			
			sumReward += reward;
			
			// set result
			policyResult->setClassificationError( i, 1-clRes );
			vector<Label>::iterator lIt;
			for( lIt=labels.begin(); lIt!=labels.end(); ++lIt )
			{
				results[lIt->idx] *= lIt->y;
			}
			
			
		}
		
		delete stateData;
		policyResult->errorRate = (AlphaReal)numErrors/(AlphaReal) numExamples;
		policyResult->avgReward = sumReward/(AlphaReal) numExamples;
		policyResult->numOfEvaluatedClassifier = (AlphaReal)overAllUsedClassifier/(AlphaReal) numExamples;
		
		return 0.0;
	}
	
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::genHeader( ofstream& out, int fnum )
	{
		out << "@RELATION mddagtrain" << endl << endl;
		for( int i=0; i<fnum; ++i )
		{
			out << "@ATTRIBUTE f" << i << " NUMERIC" << endl;
		}
		out << endl << endl;
		out << "@ATTRIBUTE class {0";
		
		for( int i=1; i < _actionNumber; ++i )
			out << ", " << i;
		out << "}" << endl << endl;
		out << "@DATA" << endl;
		out << flush;
		
		return 0.0;
	}
	
	// -------------------------------------------------------------------------
	void MDDAGLearner::getStateVector( vector<FeatureReal>& state, int iter, vector<AlphaReal>& margins )
	{
		
		if ( _inBaseLearnerName.compare( "HaarSingleStumpLearner" ) == 0)
		{									
			
			int classNum = margins.size();
			AlphaReal sumOfPosterios = 0.0;
			vector<AlphaReal> posteriors( margins.size() );
			
			sumOfPosterios = getNormalizedScores( margins, posteriors, iter );
			
#ifdef _ADD_SUMOFSCORES_TO_STATESPACE_			
			state.resize(classNum+1);
			state[classNum] = sumOfPosterios; 
			//state[classNum+1] = iter; 
#else			
			state.resize(classNum+4);			
#endif			
			
			for(int l=0; l<classNum; ++l )
				state[l] = margins[l];					
				//state[l] = posteriors[l];					
			
//			
//			HaarSingleStumpLearner* bLearner = dynamic_cast<HaarSingleStumpLearner*> (_foundHypotheses[iter]);	
//			nor_utils::Rect& rect = bLearner->getSelectedConfig();
//			state[classNum] = rect.x;
//			state[classNum+1] = rect.y;
//			state[classNum+2] = rect.width; 
//			state[classNum+3] =  rect.height;
		} else if ( _inBaseLearnerName.compare( "SingleStumpLearner" ) == 0)
		{
			int classNum = margins.size();
			AlphaReal sumOfPosterios = 0.0;
			vector<AlphaReal> posteriors( margins.size() );
			
			sumOfPosterios = getNormalizedScores( margins, posteriors, iter );
			
#ifdef _ADD_SUMOFSCORES_TO_STATESPACE_			
			state.resize(classNum+1);
			state[classNum] = sumOfPosterios; 
			//state[classNum+1] = iter; 
#else			
			state.resize(classNum);			
#endif			
			
			for(int l=0; l<classNum; ++l )
				state[l] = margins[l];					
			//state[l] = posteriors[l];					
		} else if ( _inBaseLearnerName.compare( "TreeLearner" ) == 0) // with SingleStumpLearner
		{
			int classNum = margins.size();
			AlphaReal sumOfPosterios = 0.0;
			vector<AlphaReal> posteriors( margins.size() );
			
			sumOfPosterios = getNormalizedScores( margins, posteriors, iter );
			
#ifdef _ADD_SUMOFSCORES_TO_STATESPACE_			
			state.resize(classNum+1);
			state[classNum] = sumOfPosterios; 
			//state[classNum+1] = iter; 
#else			
			state.resize(classNum);			
#endif			
			
			for(int l=0; l<classNum; ++l )
				state[l] = margins[l];					
			//state[l] = posteriors[l];					
		} else {
			cout << "State definition is not implemented!" << endl;
			exit(-1);
		}
		
		
	}
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::getNormalizedScores( vector<AlphaReal>& scores, vector<AlphaReal>& normalizedScores, int iter )
	{
		normalizedScores.resize(scores.size());
		if (iter==0) 
		{			
			fill( normalizedScores.begin(), normalizedScores.end(), 0.0 );
			return 0.0;
		}
		if (scores.size()<=2)
		{			
			copy( scores.begin(), scores.end(), normalizedScores.begin() );
			return 0.0;
		}
		
		const int classNum = scores.size();
		AlphaReal sumOfMargins = 0.0;
		for ( int i=0; i<classNum; ++i ) 
		{	
			normalizedScores[i] = scores[i];
			sumOfMargins += abs(scores[i]);
		}
		
		if ( ! nor_utils::is_zero( sumOfMargins ) )
		{
			for ( int i=0; i<classNum; ++i ) 
			{
				normalizedScores[i] /= sumOfMargins;		
				//				if ( posteriors[i] != posteriors[i])
				//				{
				//					cout << "NaN" << endl;
				//				}
			}
			
		}
		
		return sumOfMargins;
	}
	
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::getReward( vector<AlphaReal>& margins, InputData* pData, int index )
	{
		AlphaReal reward=0.0;
		
		vector<Label>& labels = pData->getLabels(index);
		vector<Label>::const_iterator lIt;
		
		AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
		AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
		AlphaReal minPosClass = numeric_limits<AlphaReal>::max();
		
		int forecastlabel = -1;
		int l;
		int allZero = 1;
		switch (_rewardtype) {
			case RW_ZEROONE:
				for(l=0, lIt=labels.begin(); lIt != labels.end(); ++lIt, ++l )
				{
					if ( margins[l] != 0 )
					{
						allZero=0;
						break;
					}
				}
				
				if (allZero==1) return 0.0;//return -1.0;
				
				maxNegClass = -numeric_limits<AlphaReal>::max();
				minPosClass = numeric_limits<AlphaReal>::max();
				
				for ( l=0,lIt = labels.begin(); lIt != labels.end(); ++lIt, ++l )
				{
					// get the negative winner class
					if ( lIt->y < 0 && margins[l] > maxNegClass )
						maxNegClass = margins[l];
					
					// get the positive winner class
					if ( lIt->y > 0 && margins[l] < minPosClass )
						minPosClass = margins[l];
				}
				
				// if the vote for the worst positive label is lower than the
				// vote for the highest negative label -> error
				if (minPosClass <= maxNegClass)
					reward=0.0;
				else 
					reward=1.0;
				
				//				for(l=0, lIt=labels.begin(); lIt != labels.end(); ++lIt, ++l )
				//				{
				//					if (margins[l]>maxMargin)
				//					{
				//						maxMargin=margins[l];
				//						forecastlabel=l;
				//					}										
				//				}
				//				if (labels[forecastlabel].y>0)
				//				{
				//					reward=1.0;
				//				} else {
				//					reward=0.0;
				//				}
				
				break;
			case RW_EXPLOSS:
				for(l=0, lIt=labels.begin(); lIt != labels.end(); ++lIt, ++l )
				{
					if ( margins[l] != 0 )
					{
						allZero=0;
						break;
					}
				}
				
				if (allZero==1) return 0.0;//return -1.0;
				
				if (margins.size()<=2) // binary classification
				{
					reward = exp(-labels[0].y*margins[0]);
				} else {
					for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
					{
						reward += exp(-labels[lIt->idx].y*(margins[lIt->idx]/_sumAlphas[_shypIter]));
					}
				}
				reward=-reward;
				break;
			case RW_MCEXPLOSS:
				for ( l=0,lIt = labels.begin(); lIt != labels.end(); ++lIt, ++l )
				{
					// get the negative winner class
					if ( lIt->y < 0 && margins[l] > maxNegClass )
						maxNegClass = margins[l]/_sumAlphas[_shypIter];
					
					// get the positive winner class
					if ( lIt->y > 0 && margins[l] < minPosClass )
						minPosClass = margins[l]/_sumAlphas[_shypIter];
				}
				
				reward = -exp( maxNegClass - minPosClass );
				
				break;
			default:
				break;
		}
		return reward;
	}
	
	
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::classify(const nor_utils::Args& args)
	{
		//		MDDAGClassifier classifier(args, _verbose);
		//		
		//		// -test <dataFile> <shypFile>
		//		string testFileName = args.getValue<string>("test", 0);
		//		string shypFileName = args.getValue<string>("test", 1);
		//		int numIterations = args.getValue<int>("test", 2);
		//		
		//		string outResFileName;
		//		if ( args.getNumValues("test") > 3 )
		//			args.getValue("test", 3, outResFileName);
		//		
		//		classifier.run(testFileName, shypFileName, numIterations, outResFileName);
	}
	
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::doConfusionMatrix(const nor_utils::Args& args)
	{
	}
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	InputData* MDDAGLearner::getRolloutData(const nor_utils::Args& args, const string fname )
	{		
		BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
		pWeakHypothesisSource->initLearningOptions(args);
		
		InputData* data = pWeakHypothesisSource->createInputData();
		data->initOptions(args);
		data->setInitWeighting( WIT_PROP_ONLY );
		
		data->load(fname, IT_TRAIN, _verbose);
		
		return data;
	}
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::doPosteriors(const nor_utils::Args& args)
	{		
		// load the arguments
		this->getArgs(args);
		
		int numofargs = args.getNumValues( "posteriors" );
		// -posteriors <dataFile> <shypFile> <outFile> <numIters>
		string testFileName = args.getValue<string>("posteriors", 0);
		string shypFileName = args.getValue<string>("posteriors", 1);
		string outFileName = args.getValue<string>("posteriors", 2);
		int numIterations = args.getValue<int>("posteriors", 3);
		int period = 0;
		
		if ( numofargs == 5 )
			period = args.getValue<int>("posteriors", 4);
		
		//////////////////////////////////////////
		// strong classifier
		//////////////////////////////////////////		
		
		// get the registered weak learner (type from name)
		_inBaseLearnerName = UnSerialization::getWeakLearnerName(shypFileName);		
		BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(_inBaseLearnerName);
		
		// initialize learning options; normally it's done in the strong loop
		// also, here we do it for Product learners, so input data can be created
		pWeakHypothesisSource->initLearningOptions(args);		
		
		// get the testing input data, and load it
		InputData* pTestData = NULL;
		pTestData = pWeakHypothesisSource->createInputData();
		pTestData->initOptions(args);
		pTestData->load(testFileName, IT_TEST, _verbose);
		
		// The class that loads the weak hypotheses
		UnSerialization us;
		
		// loads them
		us.loadHypotheses(shypFileName, _foundHypotheses, pTestData);
		_foundHypotheses.resize(numIterations);
		_shypIter=numIterations;
		
		//////////////////////////////////////////
		// policy
		//////////////////////////////////////////		
		string policyShypFileName("");
		
		if ( args.hasArgument("shypname") )
			args.getValue("shypname", 0, policyShypFileName);
		else
			policyShypFileName = string(SHYP_NAME);
		
		policyShypFileName = nor_utils::addAndCheckExtension(policyShypFileName, SHYP_EXTENSION);
		
		
		char tmpFileNameChar[4096];
		string rolloutDataFile;
		
		
		// load the policy 
		
		_policy = ClassificationBasedPolicyFactory::getPolicyObject(args, _actionNumber);
		InputData* rolloutTrainingData;
		
		if (_verbose>0)
			cout << "Rollout...";
		
		sprintf( tmpFileNameChar, "tmp.txt" );
		rolloutDataFile = _outDir + tmpFileNameChar;
		
		// because of the class map
		rollout( args, pTestData, rolloutDataFile, 100 );
		rolloutTrainingData = getRolloutData( args, rolloutDataFile );
		
		if (_verbose>0)
			cout << "Done." << endl;
		
		if (_verbose>0)
			cout << "Loading policy from " << policyShypFileName << "...";
		
		_policy->load(policyShypFileName, rolloutTrainingData);
		
		if (_verbose>0)
			cout << "Done." << endl;
		
		//////////////////////////////////////////
		// posteriors
		//////////////////////////////////////////	
		PolicyResult* policyResultTest = new PolicyResult(pTestData );
		getErrorRate(pTestData, outFileName.c_str(), policyResultTest);
		if ( !_outputInfoFile.empty() ) 
		{
			
			_outStream.open(_outputInfoFile.c_str());
			
			// is it really open?
			if ( !_outStream.is_open() )
			{
				cerr << "ERROR: cannot open the output steam (<" 
				<< _outputInfoFile << ">) for the step-by-step info!" << endl;
				exit(1);
			}			
		}
		_outStream << "Error" << "\t" << "Evaluated" << "\t" << "Avg. Rew." << "\t" << endl;
		_outStream << policyResultTest->errorRate << "\t" << policyResultTest->numOfEvaluatedClassifier << "\t" << policyResultTest->avgReward << "\t";
		_outStream << endl << flush;
		
		cout << "Error" << "\t" << "Evaluated" << "\t" << "Avg. Rew." << "\t" << endl;
		cout << policyResultTest->errorRate << "\t" << policyResultTest->numOfEvaluatedClassifier << "\t" << policyResultTest->avgReward << "\t";
		cout << endl << flush;
		
		
		_outStream.close();
	}
	
	// -------------------------------------------------------------------------
	
	int MDDAGLearner::resumeProcess(const nor_utils::Args& args, InputData* pTestData)
	{
		int numPolicies = 0;
		_policy = ClassificationBasedPolicyFactory::getPolicyObject(args, _actionNumber);
		
		if (_resumeShypFileName.empty())
			return numPolicies;
		
		char tmpFileNameChar[4096];
		string rolloutDataFile;
		
		InputData* rolloutTrainingData;		
		sprintf( tmpFileNameChar, "tmp.txt" );
		rolloutDataFile = _outDir + tmpFileNameChar;
		
		// because of the class map
		rollout( args, pTestData, rolloutDataFile, 100 );
		rolloutTrainingData = getRolloutData( args, rolloutDataFile );
		
		if (_verbose>0)
			cout << "Loading policy from " << _resumeShypFileName << "...";
		
		_policy->load(_resumeShypFileName, rolloutTrainingData);
		
		numPolicies = static_cast<AdaBoostPolicyArray*>(_policy)->getNumOfPolicies();
		
		if (_verbose > 0)
			cout << "Done!" << endl;
		
		if (_verbose > 0)
			cout << "Number of policies" << numPolicies << endl << flush;
		
		return numPolicies;
		
	}
	
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------	
	void MDDAGLearner::computeResults(InputData* pData, vector<BaseLearner*>& weakHypotheses, vector< ExampleResults* >& results )
	{
		assert( !weakHypotheses.empty() );
		
		const int numClasses = pData->getNumClasses();
		const int numExamples = pData->getNumExamples();
		
		// Creating the results structures. See file Structures.h for the
		// PointResults structure
		results.clear();
		results.reserve(numExamples);
		for (int i = 0; i < numExamples; ++i)
			results.push_back( new ExampleResults(i, numClasses) );
		
		// iterator over all the weak hypotheses
		vector<BaseLearner*>::const_iterator whyIt;
		int t;
		
		// for every feature: 1..T
		for (whyIt = weakHypotheses.begin(), t = 0; 
			 whyIt != weakHypotheses.end() && t < weakHypotheses.size(); ++whyIt, ++t)
		{
			BaseLearner* currWeakHyp = *whyIt;
			AlphaReal alpha = currWeakHyp->getAlpha();
			
			// for every point
			for (int i = 0; i < numExamples; ++i)
			{
				// a reference for clarity and speed
				vector<AlphaReal>& currVotesVector = results[i]->getVotesVector();
				
				// for every class
				for (int l = 0; l < numClasses; ++l)
					currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
			}
		}
		
	}
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::getClassError( InputData* pData, const vector<ExampleResults*>& results, AlphaReal& classError)
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
		int numErrors = 0;
		
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
            vector<Label>::const_iterator lIt;
			
            // the vote of the winning negative class
            AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
            // the vote of the winning positive class
            AlphaReal minPosClass = numeric_limits<AlphaReal>::max();
			
            vector<AlphaReal>& currVotesVector = results[i]->getVotesVector();
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the negative winner class
                if ( lIt->y < 0 && currVotesVector[lIt->idx] > maxNegClass )
                    maxNegClass = currVotesVector[lIt->idx];
                
                // get the positive winner class
                if ( lIt->y > 0 && currVotesVector[lIt->idx] < minPosClass )
                    minPosClass = currVotesVector[lIt->idx];
            }
            
            // if the vote for the worst positive label is lower than the
            // vote for the highest negative label -> error
            if (minPosClass <= maxNegClass)
                ++numErrors;
        }
        
        
        // The error is normalized by the number of points
        classError =  (AlphaReal)(numErrors)/(AlphaReal)(numExamples);
	}
	
	// -------------------------------------------------------------------------
			
}

