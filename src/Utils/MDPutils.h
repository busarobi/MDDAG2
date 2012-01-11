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
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------
	class GenericClassificationBasedPolicy
	{
	public:
		GenericClassificationBasedPolicy(const nor_utils::Args& args) : _args( args )
		{
		}
		
		
		// abstract functions 
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations ) = 0;		
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution ) = 0;
		
		virtual int getNextAction( InputData* state );
		virtual void save( const string fname ) = 0;
	protected:
		int _actionNum;
		const nor_utils::Args& _args;
		string _baseLearnerName;
	};
	
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------
	class AdaBoostPolicy : public GenericClassificationBasedPolicy
	{		
	public:
		AdaBoostPolicy(const nor_utils::Args& args) : GenericClassificationBasedPolicy(args) {}
		
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations );
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution );
		virtual void save( const string fname );
		
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
		AdaBoostPolicyArray(const nor_utils::Args& args, const AlphaReal alpha ) : GenericClassificationBasedPolicy(args), _coefficients(0), _policies( 0 ), _alpha(alpha) {}
		
		virtual AlphaReal trainpolicy( InputData* pTrainingData, const string baseLearnerName, const int numIterations );
		virtual void getDistribution( InputData* state, vector<AlphaReal>& distribution );
		virtual void save( const string fname );
		
	protected:
		vector< AdaBoostPolicy* >	_policies;
		AlphaReal					_alpha;
		vector< AlphaReal >			_coefficients;
	};
	
	
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------	
    class ClassificationBasedPolicyFactory
	{
	public:
		static GenericClassificationBasedPolicy* getPolicyObject( const nor_utils::Args& args )
		{
			GenericClassificationBasedPolicy* policy;
			AlphaReal policyAlpha = 0.0;
			
			if ( args.hasArgument("policyalpha") )
				args.getValue("policyalpha", 0, policyAlpha);
			
			if ( nor_utils::is_zero( policyAlpha ) )
			{				
				policy = new AdaBoostPolicy(args);
			}
			else
			{
				policy = new AdaBoostPolicyArray(args, policyAlpha);
			}
				
			return policy;
		}
	};	
	//-------------------------------------------------------------------
	//-------------------------------------------------------------------

} // end of namespace MultiBoost

#endif // __MDDAG_H
