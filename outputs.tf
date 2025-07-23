output "vpc_id" {
  value = module.vpc.vpc_id
}

output "public_subnet_id" {
  value = module.vpc.public_subnet_id
}

output "instance_id" {
  value = module.ec2.instance_id
}

output "instance_public_ip" {
  value = module.ec2.public_ip
}

output "private_subnet_id" {
  value = module.vpc.private_subnet_id
}

output "instance_id_private" {
  value = module.ec2_private.instance_id
}

output "instance_private_public_ip" {
  value = module.ec2_private.public_ip
}
