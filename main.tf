provider "aws" {
  region = "ap-south-1"
}
# temp comment to trigger workflow

module "vpc" {
  source              = "./modules/vpc"
  vpc_cidr            = var.vpc_cidr
  vpc_name            = var.vpc_name
  public_subnet_cidr  = var.public_subnet_cidr
  public_subnet_name  = var.public_subnet_name
  availability_zone   = var.availability_zone
  private_subnet_cidr = var.private_subnet_cidr
  private_subnet_name = var.private_subnet_name
  private_subnet_az   = var.private_subnet_az
}

module "ec2" {
  source         = "./modules/ec2"
  ami_id         = var.ami_id
  instance_type  = var.instance_type
  subnet_id      = module.vpc.public_subnet_id
  instance_name  = var.instance_name
}

module "ec2_private" {
  source         = "./modules/ec2"
  ami_id         = var.ami_id
  instance_type  = var.instance_type
  subnet_id      = module.vpc.private_subnet_id
  instance_name  = var.instance_name_private
}
