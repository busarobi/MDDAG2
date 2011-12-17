/*
 *  GoldenSection.h
 *  Multiboost.svn
 *
 *  Created by Djalel Benbouzid on 11/04/11.
 *  Copyright 2011 AppStat Group. All rights reserved.
 *
 */

#ifndef __GOLDEN_SECTION_H
#define __GOLDEN_SECTION_H

#include <cmath>
#include <StrongLearners/FCBoostLearner.h>

using namespace MultiBoost;
using namespace std;


namespace num_recipes {
    
template<class T>
inline const T &MAX(const T &a, const T &b)
{return b > a ? (b) : (a);}

inline double MAX(const double &a, const double &b)
{return b > a ? (b) : double(a);}
    
template<class T>
inline T SIGN(const T &a, const T &b)
{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

inline double SIGN(const double &a, const double &b)
{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

template<class T>
inline void SWAP(T &a, T &b)
{T dum=a; a=b; b=dum;}


class BaseLearner;
class InputData;
    
struct Bracketmethod {

    double ax,bx,cx,fa,fb,fc; 
    
    template <class T> 
    void bracket(const double a, const double b, T &func)
    {
        const double GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20;

        ax=a; 
        bx=b; 
        double fu;

        fa=func(ax); 
        fb=func(bx); 
        
        if (fb > fa) {
            SWAP(ax,bx); 
            SWAP(fb,fa);
        } 
        cx=bx+GOLD*(bx-ax); 
        fc=func(cx); 
        
        while (fb > fc) {
            double r=(bx-ax)*(fb-fc); 
            double q=(bx-cx)*(fb-fa); 
            double u=bx-((bx-cx)*q-(bx-ax)*r)/
                (2.0*SIGN(MAX(abs(q-r),TINY),q-r)); 
            double ulim=bx+GLIMIT*(cx-bx);

            if ((bx-u)*(u-cx) > 0.0) { 
                fu=func(u);
                if (fu < fc) { 
                    ax=bx;
                    bx=u; 
                    fa=fb; 
                    fb=fu; 
                    return;
                } else if (fu > fb) { 
                    cx=u;
                    fc=fu; 
                    return;
                } 
                u=cx+GOLD*(cx-bx); 
                fu=func(u);
            } else if ((cx-u)*(u-ulim) > 0.0) {
                fu=func(u);	
                
                if (fu < fc) { 
                    shft3(bx,cx,u,u+GOLD*(u-cx)); 
                    shft3(fb,fc,fu,func(u));
                } 
            } else if ((u-ulim)*(ulim-cx) >= 0.0) {
                u=ulim;
                fu=func(u); 
            } else {
                u=cx+GOLD*(cx-bx);
                fu=func(u);
            } 
            shft3(ax,bx,cx,u); 
            shft3(fa,fb,fc,fu);
        }
    }
    
    inline void shft2(double &a, double &b, const double c) {
        a=b; b=c;
    } 
    inline void shft3(double &a, double &b, double &c, const double d) {
        a=b; b=c; c=d;
    } 
    inline void mov3(double &a, double &b, double &c, const double d, const double e, const double f) {
        a=d; b=e; c=f;
    } 
};

//template <typename T>
struct Golden : Bracketmethod {
    double xmin,fmin; 
    const double tol; 
    
    int numIterations;
    
    Golden(const double toll=3.0e-8) : tol(toll), numIterations(0) {} 
    
    template <typename T> double minimize(T &func)    
    {
        const double R=0.61803399,C=1.0-R; 
        double x1,x2; 
        double x0=ax; 
        double x3=cx;
        if (abs(cx-bx) > abs(bx-ax)) { 
            x1=bx;
            x2=bx+C*(cx-bx); 
        } else {
            x2=bx; 
            x1=bx-C*(bx-ax);    
        } 
        double f1=func(x1); 
        double f2=func(x2); 
        
        
        while (abs(x3-x0) > tol*(abs(x1)+abs(x2))) {
            
            ++numIterations;
            if (f2 < f1) { 
                shft3(x0,x1,x2,R*x2+C*x3); 
                shft2(f1,f2,func(x2));
            } else { 
                shft3(x3,x2,x1,R*x1+C*x0); shft2(f2,f1,func(x1));
            }
        }
        
        if(f1<f2){
            xmin=x1;
            fmin=f1; 
        } else {
            xmin=x2; 
            fmin=f2;
        } 
        return xmin;
    }


    template <typename T> double minimize(T &func, MultiBoost::InputData* pData, int stageNumber, MultiBoost::BaseLearner* pWeakHypothesis, int positiveLabelIndex)    
    {
        const double R=0.61803399,C=1.0-R; 
        double x1,x2; 
        double x0=ax; 
        double x3=cx;
        if (abs(cx-bx) > abs(bx-ax)) { 
            x1=bx;
            x2=bx+C*(cx-bx); 
        } else {
            x2=bx; 
            x1=bx-C*(bx-ax);    
        } 
        double f1=func(x1, pData, stageNumber, pWeakHypothesis, positiveLabelIndex); 
        double f2=func(x2, pData, stageNumber, pWeakHypothesis, positiveLabelIndex); 
        
        
        while (abs(x3-x0) > tol*(abs(x1)+abs(x2))) {
            
            ++numIterations;
            if (f2 < f1) { 
                shft3(x0,x1,x2,R*x2+C*x3); 
                shft2(f1,f2,func(x2, pData, stageNumber, pWeakHypothesis, positiveLabelIndex));
            } else { 
                shft3(x3,x2,x1,R*x1+C*x0); shft2(f2,f1,func(x1, pData, stageNumber, pWeakHypothesis, positiveLabelIndex));
            }
        }
        
        if(f1<f2){
            xmin=x1;
            fmin=f1; 
        } else {
            xmin=x2; 
            fmin=f2;
        } 
        return xmin;
    }
    
};
    
    
}// end of namespace
    
#endif // __GOLDEN_SECTION_H
    
