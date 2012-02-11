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
#include "MultiMDDAGLearner.h"


#define _ADD_SUMOFSCORES_TO_STATESPACE_

namespace MultiBoost {
	// -----------------------------------------------------------------------------------
	
	void MultiMDDAGLearner::getArgs(const nor_utils::Args& args)
	{
		MDDAGLearner::getArgs(args);

		// Set the value of theta
		if ( args.hasArgument("updateperc") )
			args.getValue("updateperc", 0, _randomNPercent);  
		
	}
	
	// -----------------------------------------------------------------------------------
	
	int MultiMDDAGLearner::resumeProcess(const nor_utils::Args& args, InputData* pTestData)
	{
		int numPolicies = 0;
		
		AlphaReal policyAlpha = 0.0;
		
		if ( args.hasArgument("policyalpha") )
			args.getValue("policyalpha", 0, policyAlpha);		
		
		_policy = new AdaBoostArrayOfPolicyArray(args, _actionNumber);
		
		return numPolicies;		
	}
	
	// -------------------------------------------------------------------------
	void MultiMDDAGLearner::parallelRollout(const nor_utils::Args& args, InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy, PolicyResult* result, const int weakLearnerPostion)
	{
		vector<AlphaReal> policyError(_shypIter);
		vector<InputData*> rollouts(_shypIter,NULL);
		
		// generate rollout
		if (_randomNPercent>0)
		{
			vector<int> randomIndices(_shypIter);
			for( int si = 0; si < _shypIter; ++si ) randomIndices[si]=si;
			random_shuffle(randomIndices.begin(), randomIndices.end());
			
			int ig = static_cast<int>(static_cast<float>(_shypIter * _randomNPercent) / 100.0);
			for( int si = 0; si < ig; ++si )
			{
				stringstream ss(fname);
				//			if (si>0)
				//			{				
				//				ss << fname << "_" << si;
				//			} else {
				//				ss << fname;
				//			}
				
				
				MDDAGLearner::parallelRollout(args, pData, ss.str(), rsize, policy, result, randomIndices[si]);
				InputData* rolloutTrainingData = getRolloutData( args, ss.str() );	
				
				if (_verbose)
					cout << "---> Rollout size("<< randomIndices[si] << ")" << rolloutTrainingData->getNumExamples() << endl;
				
				rollouts[randomIndices[si]] = rolloutTrainingData;
			}
			
		} else {			
			for( int si = 0; si < _shypIter; ++si )
			{
				stringstream ss(fname);
				//			if (si>0)
				//			{				
				//				ss << fname << "_" << si;
				//			} else {
				//				ss << fname;
				//			}
				
				
				MDDAGLearner::parallelRollout(args, pData, ss.str(), rsize, policy, result, si);
				InputData* rolloutTrainingData = getRolloutData( args, ss.str() );	
				
				if (_verbose)
					cout << "---> Rollout size("<< si << ")" << rolloutTrainingData->getNumExamples() << endl;
				
				rollouts[si] = rolloutTrainingData;
			}
		}
		// update policy
		int numOfUpdatedPolicy = 0;
		for( int si = 0; si < _shypIter; ++si )
		{			
			if ((rollouts[si]==NULL) || (rollouts[si]->getNumExamples()<=2)) continue;				
			policyError[si] = _policy->trainpolicy( rollouts[si], _baseLearnerName, _trainingIter, si );			
			numOfUpdatedPolicy++;
		}
		
		if (_verbose)
			cout << "--> Number of updated policy" << numOfUpdatedPolicy << endl << flush;
		
		//release rolouts
		for( int si = 0; si < _shypIter; ++si )
		{
			delete rollouts[si];
		}		
	}		
	// -------------------------------------------------------------------------	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------	
	// -------------------------------------------------------------------------
	
}


