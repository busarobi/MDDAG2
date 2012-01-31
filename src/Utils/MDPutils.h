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


#ifndef __MDP_UTILS_H
#define __MDP_UTILS_H

#include "Defaults.h"
#include "IO/InputData.h"
#include "WeakLearners/BaseLearner.h"
#include "StrongLearners/AdaBoostMHLearner.h"
#include "Utils/Utils.h"

#include <vector>


using namespace std;

namespace MultiBoost {
	class ClassificationBasedPolicyFactory;
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------
	class GenericClassificationBasedPolicy
	{
	public:
		GenericClassificationBasedPolicy(const nor_utils::Args& args, const int actionNumber, const string ID) : _args( args ), _ID( ID )
		{
			_actionNum = actionNumber;
			if ( args.hasArgument("verbose") )
				args.getValue("verbose", 0, _verbose);

		}
		
		
		// abstract functions 
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations ) = 0;		
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution ) = 0;
		virtual void getExplorationDistribution(InputData* state, vector<AlphaReal>& distribution )
		{
			getDistribution( state, distribution );
		}
		
		virtual int getNextAction( InputData* state );
		virtual int getExplorationNextAction( InputData* state );
		
		// IO
		virtual void save( const string fname, InputData* pData = NULL ) = 0;
		virtual int  load( const string fname, InputData* pData ) = 0;
		
		// 2 or 3
		virtual void setActionNumber( int actionNumber ) { _actionNum = actionNumber; }
		
		virtual const string& getID() { return _ID; }
	protected:
		int _actionNum;
		const nor_utils::Args& _args;
		string _baseLearnerName;
		int _verbose;
		const string _ID;
	};
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------
	class RandomPolicy : public GenericClassificationBasedPolicy
	{
	public:
		RandomPolicy(const nor_utils::Args& args, const int actionNumber ) : GenericClassificationBasedPolicy(args, actionNumber, "random" ) {}
		
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations ) {}
		
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution )
		{
			distribution.resize(_actionNum);
			for (int i=0; i<_actionNum; ++i )
			{
				distribution[i] = 1.0 / static_cast<AlphaReal> (_actionNum);
			}
		}
		
		virtual int getNextAction( InputData* state )
		{
			int action = rand() % _actionNum;
			return action;
		}
		
		virtual void save( const string fname, InputData* pData = NULL ) 
		{
			ofstream ofile( fname.c_str() );
			if (!ofile.is_open() )
			{
				cout << "error at output random policy!" << endl;
				exit(-1);
			}
			ofile << "<random></random>" << endl;
			
			ofile.close();
		}
		
		virtual int load( const string fname, InputData* pData ) {} 
	};
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------
	class FullEvalPolicy : public GenericClassificationBasedPolicy
	{
	public:
		FullEvalPolicy(const nor_utils::Args& args, const int actionNumber ) : GenericClassificationBasedPolicy(args, actionNumber, "fulleval" ) {}
		
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations ) {}
		
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution )
		{
			distribution.resize(_actionNum);
			fill(distribution.begin(), distribution.end(), 0.0 );
			distribution[0]=1.0;
		}
		
		virtual int getNextAction( InputData* state ) { return 0; }
		
		virtual void save( const string fname, InputData* pData = NULL ) {}
		
		virtual int load( const string fname, InputData* pData ) {} 
	};		
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------
	class AdaBoostPolicy : public GenericClassificationBasedPolicy
	{		
	public:
		AdaBoostPolicy(const nor_utils::Args& args, const int actionNumber) : GenericClassificationBasedPolicy(args,actionNumber, "adaboost" ) {}
		AdaBoostPolicy(const nor_utils::Args& args, const int actionNumber, vector<BaseLearner*>& baselearners ) : GenericClassificationBasedPolicy(args,actionNumber, "adaboost") 
		{
			_weakhyp.resize(baselearners.size());
			copy(baselearners.begin(), baselearners.end(), _weakhyp.begin() );
		}
		
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations );
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution );
		
		virtual void save( const string fname, InputData* pData = NULL );
		virtual int  load( const string fname, InputData* pData );
		
		virtual int getBaseLearnerNum() { return _weakhyp.size(); }		
		virtual BaseLearner* getithBaseLearner( int i ) { return _weakhyp[i]; }
	protected:
		vector<BaseLearner*> _weakhyp;		
	};
	
	
	
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------		
	class AdaBoostPolicyArray : public GenericClassificationBasedPolicy
	{		
	public:
		AdaBoostPolicyArray(const nor_utils::Args& args, const AlphaReal alpha, const int actionNumber ) : 
				GenericClassificationBasedPolicy(args,actionNumber, "adaboostarray"), _coefficients(0), _policies( 0 ), _alpha(alpha) 
		{
			GenericClassificationBasedPolicy* fullEvalPolicy = new FullEvalPolicy(args,actionNumber);
			_policies.push_back( fullEvalPolicy );
			_coefficients.push_back( 1.0 );
		}
		
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations );
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution );
		virtual void getExplorationDistribution( InputData* state, vector<AlphaReal>& distribution );
		
		virtual void save( const string fname, InputData* pData = NULL );
		virtual int  load( const string fname, InputData* pData );
		
	protected:
		vector< GenericClassificationBasedPolicy* >	_policies;
		AlphaReal					_alpha;
		vector< AlphaReal >			_coefficients;
	};
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------	
    class ClassificationBasedPolicyFactory
	{
	public:
		static GenericClassificationBasedPolicy* getPolicyObject( const nor_utils::Args& args, const int actionNumber )
		{
			GenericClassificationBasedPolicy* policy;
			AlphaReal policyAlpha = 0.0;
			
			if ( args.hasArgument("policyalpha") )
				args.getValue("policyalpha", 0, policyAlpha);
			
			if ( nor_utils::is_zero( policyAlpha ) )
			{				
				//policy = new AdaBoostPolicy(args);
				policy = new AdaBoostPolicyArray(args, policyAlpha, actionNumber);
			}
			else
			{
				policy = new AdaBoostPolicyArray(args, policyAlpha, actionNumber);
			}
				
			return policy;
		}
		static GenericClassificationBasedPolicy* getRandomPolicyObject(const nor_utils::Args& args, const int actionNumber)
		{
			GenericClassificationBasedPolicy* policy = new RandomPolicy( args, actionNumber);
			return policy;
		}
	};	
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------

} // end of namespace MultiBoost

#endif // __MDDAG_H
