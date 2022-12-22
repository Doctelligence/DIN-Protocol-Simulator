#Author:Paritosh Ramanan
#Date:3/11/2018

$argnum=$#ARGV + 1;
if($argnum!=3){
	print "Format is <username>,</path/to/remote/folder>,<machine_name>\n";
	print "example gburdell3,~/data/remote_code_repo/,login-hive.pace.gatech.edu\n";
	exit;}

printf "Prepare $ARGV[0]\@$ARGV[2]:$ARGV[1]\n";
`rsync -r --exclude-from '.exclude_list.txt' . $ARGV[0]\@$ARGV[2]:$ARGV[1]`;