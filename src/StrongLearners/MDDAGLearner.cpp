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
		
		
		string rollouttype = "";
		if ( args.hasArgument( "rollouttype" ) )
			args.getValue("rollouttype", 0, rollouttype );   
		
		if ( rollouttype.compare( "montecarlo" ) == 0 )
			_rolloutType = RL_MONTECARLO;
		else if ( rollouttype.compare( "szatymaz" ) == 0 )
			_rolloutType = RL_SZATYMAZ;
		else {
			//cerr << "Unknown update rule in ProductLearnerUCT (set to default [logedge]" << endl;
			_rolloutType = RL_MONTECARLO;
		}
		
		if ( args.hasArgument( "outdir" ) )
			args.getValue("outdir", 0, _outDir );   
	}
	
	// -----------------------------------------------------------------------------------
	
	void MDDAGLearner::run(const nor_utils::Args& args)
	{
		// load the arguments
		this->getArgs(args);
		
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
		_outStream << "Training error: " << trainError << endl;
		if (_verbose > 0 )
			cout << "Training error: " << trainError << endl;
		
		// where the results go
		vector< ExampleResults* > testresults;
		
		// get test the results
		computeResults( pTestData, _foundHypotheses, testresults );
		
		AlphaReal testError;
		getClassError( pTestData, testresults, testError);
		_outStream << "Test error: " << testError << endl;
		if (_verbose > 0 )
			cout << "Test error: " << testError << endl;
		
		if (_verbose > 0)
			cout << "Done." << endl << flush;
		
		
		if (_verbose>0)
			cout << "Iteration number of input model:\t" << _shypIter << endl;
		
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
		_outStream << trainError << "\t" << testError << endl << flush;
		//cout << "Before serialization" << endl;
		// reload the previously found weak learners if -resume is set. 
		// otherwise just return 0
		
		
		
		Serialization ss(_shypFileName );
		ss.writeHeader(_baseLearnerName); // this must go after resumeProcess has been called
		
		
		if (_verbose == 1)
			cout << "Learning in progress..." << endl;
		
		///////////////////////////////////////////////////////////////////////
		// Starting the MDDAG main loop
		///////////////////////////////////////////////////////////////////////
		AlphaReal policyError = 0.0;
		AlphaReal trainClassificationError = 0.0;
		AlphaReal testClassificationError = 0.0;
		AlphaReal usedClassifierTrain = 0.0;
		AlphaReal usedClassifierTest = 0.0;
		
		char outfilename[4096];
		string tmpFileName;
		cout << "********************************* 0. **********************************" << endl;
		
		string rolloutDataFile = _outDir + "tmp.txt";
		_policy = new AdaBoostPolicy();
		InputData* rolloutTrainingData;
		
		if (_verbose>0)
			cout << "Rollout..." << endl;
		rollout( pTrainingData, rolloutDataFile );
		rolloutTrainingData = getRolloutData( args, rolloutDataFile );
		
		policyError = _policy->trainpolicy( args, rolloutTrainingData, _baseLearnerName, _trainingIter );
		delete rolloutTrainingData;
		
		if (_verbose>0)
			cout << "Classifying training." << endl;
		sprintf( outfilename, "outtrain_%d.txt", 0 );
		tmpFileName = _outDir + outfilename;
		getErrorRate(pTrainingData, tmpFileName.c_str(), trainClassificationError, usedClassifierTrain);
		
		if (_verbose>0)
			cout << "Classifying test." << endl;
		sprintf( outfilename, "outtest_%d.txt", 0 );
		tmpFileName = _outDir + outfilename;
		getErrorRate(pTestData,tmpFileName.c_str(),testClassificationError,usedClassifierTest);
		
		_outStream << "0\t" << policyError << "\t" << trainClassificationError << "\t" << testClassificationError << "\t";
		_outStream << usedClassifierTrain << "\t" << usedClassifierTest << endl << flush;
		cout << "Policy training error:\t" << policyError << endl;
		cout << "Error (train/test):\t" << trainClassificationError << "\t" << testClassificationError << endl;
		cout << "Num of evaluated BL (train/test):\t" << usedClassifierTrain << "\t" << usedClassifierTest << endl << flush;
		cout << "result filename: " << outfilename << endl;
		
		for (int t = 0; t < _numIterations; ++t)
		{
			cout << "*********************************" << (t+1) << ". **********************************" << endl;
			if (_verbose>0)
				cout << "Rollout..." << endl;			
			rollout( pTrainingData, rolloutDataFile, _policy );
			rolloutTrainingData = getRolloutData( args, rolloutDataFile );						
			policyError = _policy->trainpolicy( args, rolloutTrainingData, _baseLearnerName, _trainingIter );
			delete rolloutTrainingData;
			
			if (_verbose>0)
				cout << "Classifying training." << endl;			
			sprintf( outfilename, "outtrain_%d.txt", t+1 );
			tmpFileName = _outDir + outfilename;
			getErrorRate(pTrainingData, tmpFileName.c_str(), trainClassificationError, usedClassifierTrain);
			
			if (_verbose>0)
				cout << "Classifying test." << endl;			
			sprintf( outfilename, "outtest_%d.txt", t+1 );
			tmpFileName = _outDir + outfilename;
			getErrorRate(pTestData, tmpFileName.c_str(), testClassificationError, usedClassifierTest);
			
			_outStream << "0\t" << policyError << "\t" << trainClassificationError << "\t" << testClassificationError << "\t";
			_outStream << usedClassifierTrain << "\t" << usedClassifierTest << endl << flush;
			cout << "Policy training error:\t" << policyError << endl;
			cout << "Error: (train/test): " << trainError << "\t" << testError << endl << flush;
			cout << "Error (train/test):\t" << trainClassificationError << "\t" << testClassificationError << endl;
			cout << "Num of evaluated BL (train/test):\t" << usedClassifierTrain << "\t" << usedClassifierTest << endl << flush;
			cout << "result filename: " << outfilename << endl;
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
	void MDDAGLearner::rollout( InputData* pData, const string fname, AdaBoostPolicy* policy )
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
		
		vector< vector<AlphaReal> > margins(_shypIter+1);		
		vector<AlphaReal> path(_shypIter);	
		vector<AlphaReal>::iterator pIt;
		
		int usedClassifier;
		int randIndex;
		int randWeakLearnerIndex;
		int action;
		int currentpathsize;
		int currentNumberOfUsedClassifier;
		
		AlphaReal finalReward;
		AlphaReal reward;
		vector<AlphaReal> estimatedRewardsForActions(_actionNumber);
		
		InputData* data = new InputData();
		Example stateExample;
		data->addExample(stateExample);
		vector<FeatureReal>& state= const_cast< vector<FeatureReal>& >((data->getExample(0)).getValues());
		
		
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
		if ( _inBaseLearnerName.compare( "HaarSingleStumpLearner" ) ==0 )
		{
			genHeader(rolloutStream, numClasses+5);
		} if ( _inBaseLearnerName.compare( "SingleStumpLearner" ) ==0 )
		{
			genHeader(rolloutStream, numClasses+3);
		}
		
		
		
		for( int rlI = 0; rlI < _rollouts; ++rlI )
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
							//								action = policy->getNextAction( data );
							//							}
							/////////////							
							//exploration
							/////////////
							
							getStateVector( state, t, margins[t] );							
							action = policy->getNextAction( data );
							
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
					randIndex = rand() % numExamples;								
					randWeakLearnerIndex = rand() % _shypIter;								
					
					fill(margins[0].begin(), margins[0].end(), 0.0 );
					path.resize(0);
					usedClassifier = 0;
					
					for( int t = 0; t < randWeakLearnerIndex; ++t )
					{
						// no quit action
						if (policy==NULL)
							action = rand() % 2;
						else {
							vector<AlphaReal> distribution(3);
							getStateVector( state, t, margins[t] );							
							policy->getDistribution(data, distribution);
							(distribution[0]>distribution[1]) ? action=0 : action=1;
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
									action = policy->getNextAction( data );							
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
						
					for( int j=0; j<state.size(); ++j )
					{
						rolloutStream << state[j] << ",";
					}
						
					rolloutStream << "{ ";
					for( int a=0; a<_actionNumber; ++a)
					{							
						rolloutStream << a << " " << estimatedRewardsForActions[a] << " "; 
					}
						
					rolloutStream << "}" << " # " << rlI;
					rolloutStream << endl;																
					break;				
				default:
					break;
			}
		}
		
		rolloutStream.close();
	}
	
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::getErrorRate(InputData* pData, const char* fname, AlphaReal& error, AlphaReal& usedClassifierOnAvg )
	{
		const int numExamples = pData->getNumExamples();		
		const int classNum = pData->getNumClasses();
		
		vector<AlphaReal> results(classNum);
		Example stateExample;
		InputData* stateData = new InputData();
		stateData->addExample(stateExample);
		
		vector<FeatureReal>& state= const_cast<vector<FeatureReal>& >(stateData->getExample(0).getValues());
		state.resize(classNum);
		
		int numErrors = 0;
		
		ofstream out;
		out.open(fname );
		
		vector<int> usedClassifier;		
		int overAllUsedClassifier = 0;
		for(int i=0; i<numExamples; ++i )				
		{
			usedClassifier.resize(0);
			fill(results.begin(),results.end(),0.0);
			for(int t=0; t<_foundHypotheses.size(); ++t)
			{
				getStateVector( state, t, results );								
				
				
				int action = _policy->getNextAction(stateData);				
				
				if (action==0)
				{
					usedClassifier.push_back(t);
					overAllUsedClassifier++;
					for( int l=0; l<classNum; ++l )
					{
						results[l] += _foundHypotheses[t]->getAlpha() * _foundHypotheses[t]->classify(pData,i,l);
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
			
			out << clRes << "  ";
			for( int l=0; l<classNum; ++l)
				out << results[l] << " ";
				
			for( int t=0; t<usedClassifier.size(); ++t)
				out << usedClassifier[t] << " ";
			out << endl << flush;
		}
		
		delete stateData;
		error = (AlphaReal)numErrors/(AlphaReal) numExamples;
		usedClassifierOnAvg	= (AlphaReal)overAllUsedClassifier/(AlphaReal) numExamples;
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
	}
	// -------------------------------------------------------------------------
	void MDDAGLearner::getStateVector( vector<FeatureReal>& state, int iter, vector<AlphaReal>& margins )
	{
		
		if ( _inBaseLearnerName.compare( "HaarSingleStumpLearner" ) == 0)
		{
			int classNum = margins.size();
			state.resize(classNum+5);
			for(int l=0; l<classNum; ++l )
				state[l] = margins[l];					
			
			
			HaarSingleStumpLearner* bLearner = dynamic_cast<HaarSingleStumpLearner*> (_foundHypotheses[iter]);	
			nor_utils::Rect& rect = bLearner->getSelectedConfig();
			state[classNum] = rect.x;
			state[classNum+1] = rect.y;
			state[classNum+2] = rect.width; 
			state[classNum+3] =  rect.height;
			state[classNum+4] = iter; 
		} else if ( _inBaseLearnerName.compare( "SingleStumpLearner" ) == 0)
		{
			int classNum = margins.size();
			state.resize(classNum+3);
			for(int l=0; l<classNum; ++l )
				state[l] = margins[l];					
			
			
			SingleStumpLearner* bLearner = dynamic_cast<SingleStumpLearner*> (_foundHypotheses[iter]);	
			state[classNum] = bLearner->getThreshold();
			state[classNum+1] = bLearner->getSelectedColumn();
			state[classNum+2] = iter; 
		}
		
		
	}
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::getReward( vector<AlphaReal>& margins, InputData* pData, int index )
	{
		AlphaReal reward;
		
		vector<Label>& labels = pData->getLabels(index);
		vector<Label>::const_iterator lIt;
		
		AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
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
				
				if (allZero==1) return -1.0;
				
				for(l=0, lIt=labels.begin(); lIt != labels.end(); ++lIt, ++l )
				{
					if (margins[l]>maxMargin)
					{
						maxMargin=margins[l];
						forecastlabel=l;
					}										
				}
				if (labels[forecastlabel].y>0)
				{
					reward=1.0;
				} else {
					reward=-1.0;
				}
				
				break;
			default:
				break;
		}
		return reward;
	}
	
	
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::classify(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		
		// -test <dataFile> <shypFile>
		string testFileName = args.getValue<string>("test", 0);
		string shypFileName = args.getValue<string>("test", 1);
		int numIterations = args.getValue<int>("test", 2);
		
		string outResFileName;
		if ( args.getNumValues("test") > 3 )
			args.getValue("test", 3, outResFileName);
		
		classifier.run(testFileName, shypFileName, numIterations, outResFileName);
	}
	
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::doConfusionMatrix(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		
		// -cmatrix <dataFile> <shypFile>
		if ( args.hasArgument("cmatrix") )
		{
			string testFileName = args.getValue<string>("cmatrix", 0);
			string shypFileName = args.getValue<string>("cmatrix", 1);
			
			classifier.printConfusionMatrix(testFileName, shypFileName);
		}
		// -cmatrixfile <dataFile> <shypFile> <outFile>
		else if ( args.hasArgument("cmatrixfile") )
		{
			string testFileName = args.getValue<string>("cmatrix", 0);
			string shypFileName = args.getValue<string>("cmatrix", 1);
			string outResFileName = args.getValue<string>("cmatrix", 2);
			
			classifier.saveConfusionMatrix(testFileName, shypFileName, outResFileName);
		}
	}
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	InputData* MDDAGLearner::getRolloutData(const nor_utils::Args& args, const string fname )
	{		
		BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
		
		InputData* data = pWeakHypothesisSource->createInputData();
		data->initOptions(args);
		data->load(fname, IT_TRAIN, _verbose);
		
		return data;
	}
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::doPosteriors(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		int numofargs = args.getNumValues( "posteriors" );
		// -posteriors <dataFile> <shypFile> <outFile> <numIters>
		string testFileName = args.getValue<string>("posteriors", 0);
		string shypFileName = args.getValue<string>("posteriors", 1);
		string outFileName = args.getValue<string>("posteriors", 2);
		int numIterations = args.getValue<int>("posteriors", 3);
		int period = 0;
		
		if ( numofargs == 5 )
			period = args.getValue<int>("posteriors", 4);
		
		classifier.savePosteriors(testFileName, shypFileName, outFileName, numIterations, period);
	}
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	AlphaReal MDDAGLearner::updateWeights(OutputInfo* pOutInfo, InputData* pData, vector<BaseLearner*>& pWeakHypothesis){
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
		
		AlphaReal Z = 0; // The normalization factor
		
		// _hy will contain the margins
		_hy.resize(numExamples);
		for ( int i = 0; i < numExamples; ++i){
			_hy[i].resize(numClasses);
			fill( _hy[i].begin(), _hy[i].end(), 0.0 );
		}
		
		
		vector<BaseLearner*>::iterator it;
		if (_verbose > 0)
			cout << ": 0%." << flush;
		
		const int numIters = static_cast<int>(_foundHypotheses.size());
		const int step = numIters < 5 ? 1 : numIters / 5;
		
		int t = 0;
		// calculate the margins ( f^{t}(x_i) ), _hy will contain
		for( it = pWeakHypothesis.begin(); it != pWeakHypothesis.end(); it++, t++ )
		{
			
			if (_verbose > 1 && (t + 1) % step == 0)
			{
				float progress = static_cast<float>(t) / static_cast<float>(numIters) * 100.0;                             
				cout << "." << setprecision(2) << progress << "%." << flush;
			}
			
			BaseLearner* pWeakHypothesis = *it;
			
			const AlphaReal alpha = pWeakHypothesis->getAlpha();
			AlphaReal hx;
			for (int i = 0; i < numExamples; ++i)
			{
				vector<Label>& labels = pData->getLabels(i);
				vector<Label>::iterator lIt;
				for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
				{
					hx = pWeakHypothesis->classify(pData, i, lIt->idx );
					_hy[i][lIt->idx] += alpha * hx; // alpha * h_l(x_i)				
					
					lIt->weight *= exp( -alpha * hx * lIt->y );
					
					Z += lIt->weight;
					
					
				}
			}
			// renormalize the weights
			for (int i = 0; i < numExamples; ++i)
			{
				vector<Label>& labels = pData->getLabels(i);
				vector<Label>::iterator lIt;
				
				for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
				{
					lIt->weight /= Z;
				}
			}
			
			//i++;
			//if ( i % 1000 == 0 ) cout << i <<endl;
		}
		
		//upload the margins 
		pOutInfo->setTable( pData, _hy );
		return 0;
	}
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	AlphaReal MDDAGLearner::updateWeights(InputData* pData, BaseLearner* pWeakHypothesis)
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
		
		const AlphaReal alpha = pWeakHypothesis->getAlpha();
		
		AlphaReal Z = 0; // The normalization factor
		
		_hy.resize(numExamples);
		for ( int i = 0; i < numExamples; ++i) {
			_hy[i].resize(numClasses);
			fill( _hy[i].begin(), _hy[i].end(), 0.0 );
		}
		// recompute weights
		// computing the normalization factor Z
		
		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				_hy[i][lIt->idx] = pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
				lIt->y;
				Z += lIt->weight * // w
				exp( 
					-alpha * _hy[i][lIt->idx] // -alpha * h_l(x_i) * y_i
					);
				// important!
				// _hy[i] must be a vector with different sizes, depending on the
				// example!
				// so it will become:
				// _hy[i][l] 
				// where l is NOT the index of the label (lIt->idx), but the index in the 
				// label vector of the example
			}
		}
		
		AlphaReal gamma = 0;
		
		// Now do the actual re-weight
		// (and compute the edge at the same time)
		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				AlphaReal w = lIt->weight;
				gamma += w * _hy[i][lIt->idx];
				//if ( gamma < -0.8 ) {
				//	cout << gamma << endl;
				//}
				// The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
				lIt->weight = w * exp( -alpha * _hy[i][lIt->idx] ) / Z;
			}
		}
		
		
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {
		//      _hy[i][l] = pWeakHypothesis->classify(pData, i, l) * // h_l(x_i)
		//                  pData->getLabel(i, l); // y_i
		
		//      Z += pData->getWeight(i, l) * // w
		//           exp( 
		//             -alpha * _hy[i][l] // -alpha * h_l(x_i) * y_i
		//           );
		//   } // numClasses
		//} // numExamples
		
		// The edge. It measures the
		// accuracy of the current weak hypothesis relative to random guessing
		
		//// Now do the actual re-weight
		//// (and compute the edge at the same time)
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {  
		//      float w = pData->getWeight(i, l);
		
		//      gamma += w * _hy[i][l];
		
		//      // The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
		//      pData->setWeight( i, l, 
		//                        w * exp( -alpha * _hy[i][l] ) / Z );
		//   } // numClasses
		//} // numExamples
		
		return gamma;
	}
	
	// -------------------------------------------------------------------------
	
	int MDDAGLearner::resumeWeakLearners(InputData* pTrainingData)
	{
		if (_resumeShypFileName.empty())
			return 0;
		
		if (_verbose > 0)
			cout << "Reloading strong hypothesis file <" << _resumeShypFileName << ">.." << flush;
		
		// The class that loads the weak hypotheses
		UnSerialization us;
		
		// loads them
		us.loadHypotheses(_resumeShypFileName, _foundHypotheses, pTrainingData, _verbose);
		
		if (_verbose > 0)
			cout << "Done!" << endl;
		
		// return the number of iterations found
		return static_cast<int>( _foundHypotheses.size() );
	}
	
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::resumeProcess(Serialization& ss, 
									 InputData* pTrainingData, InputData* pTestData, 
									 OutputInfo* pOutInfo)
	{
		
		if (_resumeShypFileName.empty())
			return;
		
		vector<BaseLearner*>::iterator it;
		int t;
		
		// rebuild the new strong hypothesis file
		for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
		{
			BaseLearner* pWeakHypothesis = *it;
			
			// append the current weak learner to strong hypothesis file,
			ss.appendHypothesis(t, pWeakHypothesis);
		}
		
		const int numIters = static_cast<int>(_foundHypotheses.size());
		const int step = numIters < 5 ? 1 : numIters / 5;
		
		if (_verbose > 0)
			cout << "Resuming up to iteration " << _foundHypotheses.size() - 1 << ": 0%." << flush;
		
		// simulate the AdaBoost algorithm for the weak learners already found
		for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
		{
			BaseLearner* pWeakHypothesis = *it;
			
			// Output the step-by-step information
			printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);
			
			// Updates the weights and returns the edge
			AlphaReal gamma = updateWeights(pTrainingData, pWeakHypothesis);
			
			if (_verbose > 1 && (t + 1) % step == 0)
			{
				float progress = static_cast<float>(t) / static_cast<float>(numIters) * 100.0;                             
				cout << "." << setprecision(2) << progress << "%." << flush;
			}
			
			// If gamma <= theta there is something really wrong.
			if (gamma <= _theta)
			{
				cerr << "ERROR!" <<  setprecision(4) << endl
				<< "At iteration <" << t << ">, edge smaller than the edge offset (theta). Something must be wrong!" << endl
				<< "[Edge: " << gamma << " < Offset: " << _theta << "]" << endl
				<< "Is the data file the same one used during the original training?" << endl;
				//          exit(1);
			}
			
		}  // loop on iterations
		
		
		if (_verbose > 0)
			cout << "Done!" << endl;
		
	}
	
	// -------------------------------------------------------------------------
	
	void MDDAGLearner::printOutputInfo(OutputInfo* pOutInfo, int t, 
									   InputData* pTrainingData, InputData* pTestData, 
									   BaseLearner* pWeakHypothesis)
	{
		
		pOutInfo->outputIteration(t);
        pOutInfo->outputCustom(pTrainingData, pWeakHypothesis);
		//		pOutInfo->outputError(pTrainingData, pWeakHypothesis);
		if (pTestData)
        {
			//			pOutInfo->outputError(pTestData, pWeakHypothesis);
            pOutInfo->separator();
            pOutInfo->outputCustom(pTestData, pWeakHypothesis);
            
        }
        
		//		pOutInfo->outputWeightedError(pTrainingData);
		//		if (pTestData)
		//			pOutInfo->outputWeightedError(pTestData);
		
		
		/*
		 pOutInfo->outputMargins(pTrainingData, pWeakHypothesis);
		 pOutInfo->outputEdge(pTrainingData, pWeakHypothesis);
		 if (pTestData)
		 pOutInfo->outputMargins(pTestData, pWeakHypothesis);
		 pOutInfo->outputMAE(pTrainingData);      
		 if (pTestData)
		 pOutInfo->outputMAE(pTestData);  
		 */
		pOutInfo->outputCurrentTime();
		pOutInfo->endLine();
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
	// -------------------------------------------------------------------------
}