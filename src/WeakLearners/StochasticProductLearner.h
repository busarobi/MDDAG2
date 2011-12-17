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
 *    Contact: : multiboost@googlegroups.com
 *
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
 *
 */


/**
 * \file ProductLearner.h The stochastic version of ProductLearner.
 * \date 24/04/2007
 */

#ifndef __STOCHASTIC_PRODUCT_LEARNER_H
#define __STOCHASTIC_PRODUCT_LEARNER_H

#include "BaseLearner.h"
#include "ProductLearner.h"
#include "StochasticLearner.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	/**
	 * A learner that loads a set of base learners, and boosts on the top of them. 
	 */
	class StochasticProductLearner : public ProductLearner, public StochasticLearner
	{
	public:
		
		
		/**
		 * The constructor. It initializes _numBaseLearners to -1
		 * \date 26/05/2007
		 */
		StochasticProductLearner() : ProductLearner(), StochasticLearner() { }
		
		/**
		 * The destructor. Must be declared (virtual) for the proper destruction of 
		 * the object.
		 */
		virtual ~StochasticProductLearner() {
		}							
		
		/**
		 * Declare weak-learner-specific arguments.
		 * adding --baselearnertype
		 * \param args The Args class reference which can be used to declare
		 * additional arguments.
		 * \date 24/04/2007
		 */
		virtual void declareArguments(nor_utils::Args& args);
		
		/**
		 * Set the arguments of the algorithm using the standard interface
		 * of the arguments. Call this to set the arguments asked by the user.
		 * \param args The arguments defined by the user in the command line.
		 * \date 24/04/2007
		 */
		virtual void initLearningOptions(const nor_utils::Args& args);
		
		/**
		 * Returns itself as object.
		 * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
		 * for the auto-registering classes.
		 * \date 14/11/2005
		 */

		virtual BaseLearner* subCreate() { return new StochasticProductLearner(); }
				
		/**
		 * Allocate memory for training.
		 * \date 03/11/2011
		 */		
		virtual void initLearning();
		
		/**
		 * Release memory usied during training.
		 * \date 03/11/2011
		 */				
		virtual AlphaReal finishLearning();
		
		/**
		 * It updates the parameter of weak learner.
		 * \return edge for the given istance
		 * \date 03/11/2011
		 */
		virtual AlphaReal update( int idx );				
		
		/**
		 * Copy all the info we need in classify().
		 * pBaseLearner was created by subCreate so it has the correct (sub) type.
		 * Usually one must copy the same fields that are loaded and saved. Don't 
		 * forget to call the parent's subCopyState().
		 * \param pBaseLearner The sub type pointer into which we copy.
		 * \see save
		 * \see load
		 * \see classify
		 * \see ProductLearner::run()
		 * \date 25/05/2007
		 */
		virtual void subCopyState(BaseLearner *pBaseLearner);
		
	protected:				
	};
	
	//////////////////////////////////////////////////////////////////////////
	
} // end of namespace MultiBoost

#endif // __STOCHASTIC_PRODUCT_LEARNER_H
