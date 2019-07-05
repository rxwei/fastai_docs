// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_09_optimizer",
products: [
.library(name: "FastaiNotebook_09_optimizer", targets: ["FastaiNotebook_09_optimizer"]),
.executable(name: "FastaiNotebook_09_optimizer_run", targets: ["FastaiNotebook_09_optimizer_run"]),

],
dependencies: [
.package(path: "../FastaiNotebook_08a_heterogeneous_dictionary")
],
targets: [
.target(name: "FastaiNotebook_09_optimizer", dependencies: ["FastaiNotebook_08a_heterogeneous_dictionary"]),
.target(name: "FastaiNotebook_09_optimizer_run", dependencies: ["FastaiNotebook_09_optimizer"]),

]
)
