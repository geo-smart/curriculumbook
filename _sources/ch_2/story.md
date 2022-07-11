# Cloud Computing: Running Python Code on AWS

## Create an AWS Educate Account
Go to [https://aws.amazon.com/education/awseducate/](https://aws.amazon.com/education/awseducate/).

Creating the account will give you access to AWS tutorials and some AWS credits.

## Create an AWS Account
Go to [https://aws.amazon.com/](https://aws.amazon.com/). You will need to ener a credit card number.

Once your account is created, you should be able to redeem your AWS credits.

## Most Important Points
- Do not forget to terminate your instances
- Check your billing dashboard on a regular basis
- The EC2 dashboard shows only instances and keys for the region you have currently selected. Other instances may be running in other regions.

### Start working! Open the home page of your AWS account
<img src="figures/AWS_home.jpg"/>

### Check the region where you are [here Oregon = us-west-2]
<img src="figures/EC2_home.jpg"/>

### You need to create a key pair to access EC2 instances
<img src="figures/EC2_home.jpg"/>

### Click on 'Create key pair' to start
<img src="figures/list_key_pairs_before.jpg"/>

### Fill in the form and click on 'Create key pair'
<img src="figures/create_key_pair.jpg"/>

### Download the file on your machine
<img src="figures/download_key_pair.jpg"/>

### Your new key pair appears on the list of available key pairs
<img src="figures/list_key_pairs_after.jpg"/>

### Move key pair to .ssh for instance
		> cd Downloads
		> mv my_key.pem ~/.ssh

### Now let us launch a new EC2 instance!
<img src="figures/EC2_home.jpg"/>

### You first need to choose an Amazon Machine Image [AMI]
<img src="figures/choose_AMI.jpg"/>

### We want one with Anaconda installed
<img src="figures/choose_anaconda_AMIs.jpg"/>

### We choose the one with Anaconda and Python 3
<img src="figures/select_anaconda_python3.jpg"/>

### Check the cost of each type of instance before choosing one
<img src="figures/list_of_prices.jpg"/>

### Select the type of instance you want: We choose t3.small
<img src="figures/choose_instance_type.jpg"/>

### Configure the instance [you can leave it as it is]
<img src="figures/configure_instance.jpg"/>

### Choose the storage [you can leave it as it is]
<img src="figures/add_storage.jpg"/>

### Add tags to remember what jobs you will be running on your EC2 instance
<img src="figures/add_tags.jpg"/>

### I want to run the Python script \textit[find\_all\_LFEs\_parallel.py
<img src="figures/enter_tag.jpg"/>

### Choose a security group [you can use the same for several instances]
<img src="figures/choose_security_group.jpg"/>

### Review your options before launching
<img src="figures/review_launch.jpg"/>

### Choose a key pair: the one you have downloaded earlier
<img src="figures/choose_key_pair.jpg"/>

### Your EC2 instance is launching
<img src="figures/ec2_is_starting.jpg"/>

### Go back to the list of instances
<img src="figures/go_back_to_instances.jpg"/>

### Your new EC2 instance is here
<img src="figures/ec2_is_here.jpg"/>

### You can modify the tags
<img src="figures/change_tag.jpg"/>

### Your can modify the security group
<img src="figures/change_security_group.jpg"/>

### Change the rules to connect to your instances
<img src="figures/add_security_rule.jpg"/>

### Your IP may have changed since the last time you connected
<img src="figures/modify_IP.jpg"/>

### Follow the instructions to connect with SSH
<img src="figures/open_terminal_and_connect.jpg"/>

### Open an SSH terminal to connect to your instance

		> cd ~/.ssh
		> ssh -i "my_key.pem" ec2-user@ec2-XXX-XXX-XXX-XXX. us-west-2.compute.amazonaws.com

### Install git and clone your repository

		> sudo yum install git
		> git clone "https://github.com/ArianeDucellier/ catalog.git"

### Create your Anaconda environment

		> cd catalog
		> conda env create -f environment.yml
		> conda activate catalog

### In my case, I just prepare some input files

		> mkdir data/response
		> cd src
		> python get_responses.py

### Use nohup to continue your computation after you disconnect

		> nohup python find_all_LFEs_parallel.py &
		> ps -x -o pid,user,%mem,command
		> exit

### Never forget to terminate your instances!!!
<img src="figures/do_not_forget_terminate.jpg"/>

### Never forget to terminate your instances!!!
<img src="figures/terminate_instance.jpg"/>

### Check if your instance has been terminated
<img src="figures/check_terminated.jpg"/>

### Go to your billing dashboard
<img src="figures/billing_dashboard.jpg"/>

### You can check how much you have been spending [do it often!]
<img src="figures/billing_home.jpg"/>

### Go to \textit[Cost Explorer"/>
<img src="figures/go_to_cost_explorer.jpg"/>

### Look at the daily amounts you spent
<img src="figures/daily_spend_view.jpg"/>

### Look at the daily amounts you spent
<img src="figures/see_history.jpg"/>

### Go to \textit[Budgets"/>
<img src="figures/go_to_budgets.jpg"/>

### You can create your own budgets
<img src="figures/create_budget.jpg"/>

## Some sources of funding

Become a member of the [Research Computing Club](https://depts.washington.edu/uwrcc/) and apply for AWS credits through their [Cloud Credit Program](https://depts.washington.edu/uwrcc/getting-started-2/cloud-computing/).

Apply to the [Integral Environmental Big Data Research Fund](https://environment.uw.edu/students/student-resources/scholarships-funding/graduate-scholarships-funding/other-graduate-funding-opportunities/integral-environmental-big-data-research-award/): For graduate students, deadline in February.

Apply to an [Azure Compute Grant](https://www.microsoft.com/en-us/ai/ai-for-earth-grants) if you want to use Azure instead of AWS, and your research focuses on Climate, Agriculture, Biodiversity, or Water.

Workshop on Azure
- Azure 101: Getting Started with Azure: [https://aka.ms/university-azure/GettingStartedAzure](https://aka.ms/university-azure/GettingStartedAzure)
- Working with Data in Azure: [https://aka.ms/university-azure/DataOnAzure](https://aka.ms/university-azure/DataOnAzure)
- Machine Learning on Azure: [https://aka.ms/university-azure/MachineLearning](https://aka.ms/university-azure/MachineLearning)
- Office Hours: April 9 at 10am PST, April 16 at 10am PST, April 23 at 10am PST

## Happy computing!
