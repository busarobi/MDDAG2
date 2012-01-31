/*
 *  MDPutils.cpp
 *  MDDAG2
 *
 *  Created by Robert Busa-Fekete on 1/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "MDPutils.h"
#include "IO/Serialization.h"

namespace MultiBoost {
	// -----------------------------------------------------------------------------------
	int GenericClassificationBasedPolicy::getNextAction( InputData* state )
	{
		vector<AlphaReal> forecast(_actionNum);
		
		// getExplorationDistribution means that the policies are aggregated
		getExplorationDistribution(state, forecast);
		
		AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
		int forecastlabel = -1;
		AlphaReal tmpVal = forecast[0];
		int allEqual = 1;
		
		for(int l=0; l<_actionNum; ++l )
		{
			if (forecast[l]>maxMargin)
			{
				maxMargin=forecast[l];
				forecastlabel=l;
			}
			
			if ( ! nor_utils::is_zero(forecast[l]-tmpVal) )
			{
				allEqual=0;
			}			
		}	
		
//		vector<FeatureReal>& values = state->getValues(0);
//		for (int tmpv=0; tmpv < values.size(); ++tmpv) cout << state->getValue(1,tmpv ) << " ";
//		cout << endl;
//		for (int tmpv=0; tmpv < forecast.size(); ++tmpv) cout << forecast[tmpv] << " ";
//		cout << endl;
		
		
		// if equal
		if (allEqual)
		{				
			forecastlabel=rand() % _actionNum;					
		} else {
			//cout << forecastlabel << endl;
		}

		
		return forecastlabel;
	}
	
	// -----------------------------------------------------------------------------------	
	int GenericClassificationBasedPolicy::getExplorationNextAction( InputData* state )
	{
		vector<AlphaReal> forecast(_actionNum);
		getExplorationDistribution(state, forecast);
		
		AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
		int forecastlabel = -1;
		AlphaReal tmpVal = forecast[0];
		int allEqual = 1;
		
		for(int l=0; l<_actionNum; ++l )
		{
			if (forecast[l]>maxMargin)
			{
				maxMargin=forecast[l];
				forecastlabel=l;
			}
			
			if ( ! nor_utils::is_zero(forecast[l]-tmpVal) )
			{
				allEqual=0;
			}			
		}	
		
		//		vector<FeatureReal>& values = state->getValues(0);
		//		for (int tmpv=0; tmpv < values.size(); ++tmpv) cout << state->getValue(1,tmpv ) << " ";
		//		cout << endl;
		//		for (int tmpv=0; tmpv < forecast.size(); ++tmpv) cout << forecast[tmpv] << " ";
		//		cout << endl;
		
		
		// if equal
		if (allEqual)
		{				
			forecastlabel=rand() % _actionNum;					
		} else {
			//cout << forecastlabel << endl;
		}
		
		
		return forecastlabel;
	}
	
	// -----------------------------------------------------------------------------------	
	AlphaReal AdaBoostPolicy::trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations )
	{
		AdaBoostMHLearner* sHypothesis = new AdaBoostMHLearner();
		sHypothesis->run(_args, pTrainingData, baseLearnerName, numIterations, _weakhyp );
		delete sHypothesis;		
		_actionNum = pTrainingData->getNumClasses();
		_baseLearnerName = baseLearnerName;
		
		const int numExamples = pTrainingData->getNumExamples();
		
		vector<AlphaReal> results(_actionNum);
		int numErrors = 0;
		for(int i=0; i<numExamples; ++i )				
		{
			fill(results.begin(),results.end(),0.0);
			for(int t=0; t<_weakhyp.size(); ++t)
			{
				for( int l=0; l<_actionNum; ++l )
				{
					results[l] += _weakhyp[t]->getAlpha() * _weakhyp[t]->classify(pTrainingData,i,l);
				}
			}
			AlphaReal maxMargin = -numeric_limits<AlphaReal>::max();
			int forecastlabel = -1;
			
			for(int l=0; l<_actionNum; ++l )
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
	
	
	// -----------------------------------------------------------------------------------
	void AdaBoostPolicy::getDistribution( InputData* state, vector<AlphaReal>& distribution )
	{
		distribution.resize(_actionNum);
		fill( distribution.begin(), distribution.end(), 0.0 );
		AlphaReal sumAlpha = 0.0;
		
		for( int t = 0; t < _weakhyp.size(); ++t )
		{
			sumAlpha += _weakhyp[t]->getAlpha();
			for( int l = 0; l<_actionNum; ++l )
			{
				distribution[l] += _weakhyp[t]->getAlpha() * _weakhyp[t]->classify(state,0,l);
			}
		}
		
		// rescale into 0-1
		for( int l = 0; l<_actionNum; ++l )
		{
			distribution[l] /= sumAlpha; // [-1;1]
			distribution[l] += 1.0;
			distribution[l] /= 2.0;
		}
		
	}
	//------------------------------------------------------------------------------------------
	AlphaReal AdaBoostPolicyArray::trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations )
	{
		_actionNum = pTrainingData->getNumClasses();
		_baseLearnerName = baseLearnerName;
		
		AdaBoostPolicy* abpolicy = new AdaBoostPolicy( _args, _actionNum );
		AlphaReal retval = abpolicy->trainpolicy(pTrainingData, baseLearnerName, numIterations );
		
		for (int i=0; i<_coefficients.size(); ++i ) _coefficients[i] *= _alpha;
				
		_policies.push_back( abpolicy );
		_coefficients.push_back( 1.0 );
		
		return retval;
	}	
	//------------------------------------------------------------------------------------------
	void AdaBoostPolicyArray::getExplorationDistribution( InputData* state, vector<AlphaReal>& distribution )
	{
		vector< AlphaReal > tmpDistribution( _actionNum );		
		fill( distribution.begin(), distribution.end(), 0.0 );
		for ( int i=_policies.size()-1; 0<=i; --i )
		{
			if (_coefficients[i] < 0.01 ) break;
			//if ((_policies.size()-i)>=5) break;
			
			_policies[i]->getDistribution( state, tmpDistribution );
			for( int l=0; l < _actionNum; ++l )
			{
				distribution[l] += (_coefficients[i]*tmpDistribution[l]);
			}
		}
	}	
	//------------------------------------------------------------------------------------------
	void AdaBoostPolicyArray::getDistribution( InputData* state, vector<AlphaReal>& distribution )
	{
		const int ind = _policies.size()-1;
		_policies[ind]->getDistribution( state, distribution );
	}	
	
	//------------------------------------------------------------------------------------------	
	void AdaBoostPolicy::save( const string fname, InputData* pData )
	{
		Serialization ss(fname, false );
		ss.writeHeader(_baseLearnerName); // this must go after resumeProcess has been called
		for (int t=0; t<_weakhyp.size(); ++t)
			ss.appendHypothesis(t, _weakhyp[t]);
		ss.writeFooter();
	}
	//------------------------------------------------------------------------------------------
	void AdaBoostPolicyArray::save( const string fname, InputData* pData )
	{
//		const int ind = _policies.size()-1;
//		AdaBoostPolicy* _lastpolicy = dynamic_cast<AdaBoostPolicy*>(_policies[ind]);
//		
//		Serialization ss(fname, false );
//		ss.writeHeader(_baseLearnerName); // this must go after resumeProcess has been called		
//		for (int t=0; t<_lastpolicy->getBaseLearnerNum(); ++t)
//			ss.appendHypothesis(t, _lastpolicy->getithBaseLearner(t));
//		ss.writeFooter();		
		
		const int policyNum = _policies.size();
		
		Serialization ss(fname, false );
		ss.writeCascadeHeader(_baseLearnerName);
		for (int i=1; i < policyNum; ++i )
		{
			AdaBoostPolicy* currentpolicy = dynamic_cast<AdaBoostPolicy*>(_policies[i]);
			
			ss.appendStageSeparatorHeader( i, currentpolicy->getBaseLearnerNum(), _coefficients[i] );						
			//ss.appendStageSeparatorFooter();
			// append the current weak learner to strong hypothesis file,
			// that is, serialize it.					
			for (int t=0 ; t < currentpolicy->getBaseLearnerNum(); ++t )
			{
				BaseLearner* currentBaseLearner = currentpolicy->getithBaseLearner(t);
				currentBaseLearner->setTrainingData(pData);
				ss.appendHypothesis(t, currentBaseLearner );						
			}
		}	
		ss.writeCascadeFooter();
	}

	//------------------------------------------------------------------------------------------	
	int AdaBoostPolicy::load( const string fname, InputData* pData )
	{
		if (fname.empty())
			return 0;
		
		if (_verbose > 0)
			cout << "Reloading policy file <" << fname << ">.." << flush;
		
		// The class that loads the weak hypotheses
		UnSerialization us;
		
		// loads them
		us.loadHypotheses(fname, _weakhyp, pData, _verbose);
		
		if (_verbose > 0)
			cout << "Done!" << endl;
		
		// return the number of iterations found
		return static_cast<int>( _weakhyp.size() );
	}
	
	//------------------------------------------------------------------------------------------
	int AdaBoostPolicyArray::load( const string fname, InputData* pData )
	{
		// The class that loads the weak hypotheses
		UnSerialization us;
		
		// Where to put the weak hypotheses
		vector<vector<BaseLearner*> > weakHypotheses(0);
		vector<AlphaReal> coefficients;        
		// loads them
		//us.loadHypotheses(shypFileName, weakHypotheses, pData);
		us.loadCascadeHypotheses(fname, weakHypotheses, coefficients, pData);
		
		
		_coefficients.resize(coefficients.size()+1);
		fill(_coefficients.begin(), _coefficients.end(), 1.0 );
		
		for ( int i=_coefficients.size()-2; i >= 0; --i)
		{
			_coefficients[i] = _coefficients[i+1]*_alpha;
		}
		
		for ( int i=0; i < weakHypotheses.size(); ++i)
		{
			AdaBoostPolicy* currentPolicy = new AdaBoostPolicy(_args, _actionNum, weakHypotheses[i] );
			_policies.push_back(currentPolicy); 
		}
		
		return weakHypotheses.size();
	}
	//------------------------------------------------------------------------------------------	
	
	
	
}