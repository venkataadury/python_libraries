#!/bin/bash

infile=$1
outpref=$2

./db2_to_mol2_teb.py $infile $outpref
mline=`grep MOLECULE -m 2 -n $outpref".0.mol2"|tail -1|cut -d":" -f1`
mline=`expr $mline - 1`
echo "$mline lines to be extracted"
if test $mline -le 3
then
	# Only one conformer
	cat $outpref".0.mol2" > $outpref".mol2"
else
	# Multiple conformers
	head -$mline $outpref".0.mol2" > $outpref".mol2"
fi
mv $outpref".0.mol2" $outpref"_allconf.mol2"
