variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "vpc_name" {
  description = "Name tag for the VPC"
  type        = string
  default     = "my-vpc"
}

variable "public_subnet_cidr" {
  description = "CIDR block for the public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "public_subnet_name" {
  description = "Name tag for the public subnet"
  type        = string
  default     = "my-public-subnet"
}

variable "availability_zone" {
  description = "Availability zone for the public subnet"
  type        = string
  default     = "ap-south-1a"
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  type        = string
  default     = "ami-0a7cf821b91bcccbc" # Amazon Linux 2 in ap-south-1
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.large"
}

variable "instance_name" {
  description = "Name tag for the EC2 instance"
  type        = string
  default     = "my-ec2-instance"
}

variable "private_subnet_cidr" {
  description = "CIDR block for the private subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "private_subnet_name" {
  description = "Name tag for the private subnet"
  type        = string
  default     = "my-private-subnet"
}

variable "private_subnet_az" {
  description = "Availability zone for the private subnet"
  type        = string
  default     = "ap-south-1b"
}

variable "instance_name_private" {
  description = "Name tag for the EC2 instance in the private subnet"
  type        = string
  default     = "my-ec2-instance-private"
}
