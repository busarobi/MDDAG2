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


#ifndef __MULTI_MDDAG_LEARNER_H
#define __MULTI_MDDAG_LEARNER_H


#include "MDDAGLearner.h"

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////							

namespace MultiBoost {
	
	class MultiMDDAGLearner : public MDDAGLearner {
    public:    
		MultiMDDAGLearner() : MDDAGLearner() {}
        /**
         * Get the needed parameters (for the strong learner) from the argumens.
         * \param The arguments provided by the command line.
         */
        void getArgs(const nor_utils::Args& args);
				
        /**
         * Resume the training using the features in _resumeShypFileName if the
         * option -resume has been specified.
         * \date 21/12/2005
         */
        virtual int resumeProcess(const nor_utils::Args& args, InputData* pTestData);
		
        
		void rollout(InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy = NULL, PolicyResult* result = NULL )
		{
			cout << "Sequential rollout is not implemented!" << endl;
		}
		
		virtual void parallelRollout(const nor_utils::Args& args, InputData* pData, const string fname, int rsize, GenericClassificationBasedPolicy* policy = NULL, PolicyResult* result = NULL, const int weakLearnerPostion = -1 );
    protected:		      		
		
		
		friend class CalculateErrorRate;
		friend class Rollout;
	};		
	// ------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------
	//////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////							
}
#endif
