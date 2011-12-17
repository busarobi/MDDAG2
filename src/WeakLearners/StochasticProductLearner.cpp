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


#include "StochasticProductLearner.h"
#include <typeinfo>

namespace MultiBoost {
	
	
	REGISTER_LEARNER(StochasticProductLearner)
	
	// -----------------------------------------------------------------------
	void StochasticProductLearner::declareArguments(nor_utils::Args& args)
	{
		ProductLearner::declareArguments(args);
		StochasticLearner::declareArguments(args);
	}
	
	// -----------------------------------------------------------------------
	void StochasticProductLearner::initLearningOptions(const nor_utils::Args& args)
	{
		ProductLearner::initLearningOptions(args);
		StochasticLearner::initLearningOptions(args);
	}
	
	// -----------------------------------------------------------------------
	
	void StochasticProductLearner::initLearning()
	{
		try{
			for( int ib = 0; ib <_numBaseLearners; ++ib )
			{
				_baseLearners[ib]->setTrainingData(_pTrainingData);
				dynamic_cast<StochasticLearner*>(_baseLearners[ib])->initLearning();
			}
		} catch ( bad_cast e ) {
			cout << "The weak learner must be StochasticLearner as well" << endl;
			cout << "StochasticProductLearner::initLearning()" << endl;
			exit(-1);
		}
	}
	
	// -----------------------------------------------------------------------	
	AlphaReal StochasticProductLearner::finishLearning()
	{
		for( int ib = 0; ib <_numBaseLearners; ++ib )
		{
			_baseLearners[ib]->setTrainingData(_pTrainingData);
			dynamic_cast<StochasticLearner*>(_baseLearners[ib])->finishLearning();
		}		
	}
	
	// -----------------------------------------------------------------------
	AlphaReal StochasticProductLearner::update( int idx )
	{
		const int numClasses = _pTrainingData->getNumClasses();
		vector<Label>& labels = _pTrainingData->getLabels(idx);
		
		// store original labels
		vector<char> origLabels(numClasses);		
		for ( int k = 0; k < numClasses; ++k )
		{
			origLabels[k]=labels[k].y;
		}
		
		// here comes the while		
		for( int ib = 0; ib <_numBaseLearners; ++ib )
		{
			vector<AlphaReal> outPutOfCurrentProduct(numClasses, 1.0);
			for( int ib2 = 0; ib2 <_numBaseLearners; ++ib2 )
			{
				for (int k=0; k<numClasses; ++k )
					outPutOfCurrentProduct[k] *= _baseLearners[ib2]->classify(_pTrainingData, idx, k );
			}
			
			
			for ( int k = 0; k < numClasses; ++k )
			{			
				AlphaReal outPutOfCurrentWeakClassifier = _baseLearners[ib]->classify(_pTrainingData, idx, k );
				AlphaReal val = origLabels[k] * outPutOfCurrentProduct[k] * outPutOfCurrentWeakClassifier; 
				labels[k].y = (val < 0.0) ? -1.0 : 1.0;
			}
			AlphaReal diff = dynamic_cast<StochasticLearner*>(_baseLearners[ib])->update(idx);
			if (_verbose>4)
			{
				//cout << "---> Diff:\t" <<  diff << endl;
			}
		}
		
		// here ends the while while
		
		//restore labels
		for ( int k = 0; k < numClasses; ++k )
		{
			labels[k].y=origLabels[k];
		}
						
		return 0.0;
	}
	
	// -----------------------------------------------------------------------
	void StochasticProductLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		ProductLearner:subCopyState(pBaseLearner);
		StochasticLearner::subCopyState(pBaseLearner);
	}
	
	
} // end of namespace MultiBoost
