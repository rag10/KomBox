#!/usr/bin/env python3
"""
Ejemplo completo del hito F3 - Restricciones implícitas en Trapezoidal

Este ejemplo demuestra todas las funcionalidades implementadas:
1. KKT puro: restricción holonómica exacta con λ
2. Baumgarte: estabilización de restricción con parámetros α, β
3. Modo mixto: combinación de ambos enfoques
4. Re-evaluación de externals a t+dt
5. Proyección pos-paso opcional
6. Robustez JFNK con allow_unused

Escenario: Sistema de dos masas con restricciones diferentes
- Masa 1: x₁ = 0 (clamp absoluto via KKT)
- Masa 2: x₂ = sen(t) (seguimiento de trayectoria via Baumgarte)
"""

import os, sys, torch
# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Imports del framework KomBox
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers_trapezoidal import TrapezoidalSolver
from kombox.core.algebraic.newton_krylov import NewtonKrylov
from kombox.blocks.mechanical import Mass1D


def create_mixed_constraint_system():
    """Crea un sistema con restricciones mixtas KKT/Baumgarte."""
    
    model = Model("f3_demo_mixed")
    
    # Dos masas independientes
    model.add_block("mass1", Mass1D(m=1.0))  # Masa con restricción KKT
    model.add_block("mass2", Mass1D(m=2.0))  # Masa con restricción Baumgarte
    
    # Externals para fuerzas
    model.connect("F1", "mass1.F")
    model.connect("F2", "mass2.F")
    
    # ===== RESTRICCIONES =====
    
    # Restricción 1: x₁ = 0 (holonómica exacta, usar KKT)
    def constraint_mass1_fixed(t, states, inbuf, model, z):
        """g₁(x₁) = x₁ = 0"""
        x1 = states["mass1"][:, 0:1]  # Posición de masa1
        return x1
    
    model.add_constraint_eq("mass1_fixed", constraint_mass1_fixed)
    
    # Restricción 2: x₂ = sin(t) (seguimiento de trayectoria, usar Baumgarte)
    def constraint_mass2_trajectory(t, states, inbuf, model, z):
        """g₂(x₂, t) = x₂ - sin(t) = 0"""
        x2 = states["mass2"][:, 0:1]  # Posición de masa2
        target = torch.sin(torch.tensor(t)).expand_as(x2)
        return x2 - target
    
    model.add_constraint_eq("mass2_trajectory", constraint_mass2_trajectory)
    
    # ===== FORCE HOOKS =====
    
    # Hook 1: Fuerza de restricción para masa1 (λ₁ × ∂g₁/∂x₁ = λ₁ × 1)
    def force_hook_mass1(t, states, inbuf, model, z, lam_i):
        """Aplica fuerza λ₁ directamente sobre puerto F de mass1"""
        return {"mass1": {"F": lam_i}}  # lam_i es (B, 1)
    
    model.add_constraint_force("mass1_fixed", force_hook_mass1)
    
    # Hook 2: Fuerza de restricción para masa2 (λ₂ × ∂g₂/∂x₂ = λ₂ × 1)
    def force_hook_mass2(t, states, inbuf, model, z, lam_i):
        """Aplica fuerza λ₂ directamente sobre puerto F de mass2"""
        return {"mass2": {"F": lam_i}}  # lam_i es (B, 1)
    
    model.add_constraint_force("mass2_trajectory", force_hook_mass2)
    
    model.build()
    return model


def run_comparison_study():
    """Ejecuta estudio comparativo de los tres modos de restricciones."""
    
    print("=== ESTUDIO COMPARATIVO F3: Modos de Restricciones ===\n")
    
    # Configuración común
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B = 1  # Batch size
    T_final = 2.0
    dt = 0.02
    steps = int(T_final / dt)
    
    # Estados iniciales (violando restricciones intencionalmente)
    x1_init = torch.tensor([[0.5]], dtype=torch.float32)  # Debería ir a 0
    x2_init = torch.tensor([[0.3]], dtype=torch.float32)  # Debería seguir sin(t)
    v_init = torch.zeros((B, 1), dtype=torch.float32)
    
    # Externals: fuerzas externas pequeñas
    def externals_fn(t, k):
        return {
            "F1": {"F": torch.tensor([[0.1 * np.sin(t)]], dtype=torch.float32)},
            "F2": {"F": torch.tensor([[0.05 * np.cos(t)]], dtype=torch.float32)}
        }
    
    results = {}
    
    # ===== MODO 1: KKT PURO =====
    print("1. Ejecutando modo KKT puro...")
    
    model_kkt = create_mixed_constraint_system()
    model_kkt.initialize(
        batch_size=B, device=device, dtype=torch.float32,
        initial_states={
            "mass1": {"x": x1_init.clone(), "v": v_init.clone()},
            "mass2": {"x": x2_init.clone(), "v": v_init.clone()}
        }
    )
    
    alg_kkt = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30, verbose=False)
    solver_kkt = TrapezoidalSolver(
        algebraic_solver=alg_kkt,
        constraint_mode="kkt"
    )
    
    sim_kkt = Simulator(model_kkt, solver=solver_kkt)
    
    # Simular
    t_history = []
    x1_kkt_history = []
    x2_kkt_history = []
    target_history = []
    
    for step in range(steps):
        sim_kkt.step(dt, externals_fn=externals_fn)
        
        t_current = sim_kkt.t
        x1 = model_kkt.states["mass1"][0, 0].item()
        x2 = model_kkt.states["mass2"][0, 0].item()
        target = np.sin(t_current)
        
        t_history.append(t_current)
        x1_kkt_history.append(x1)
        x2_kkt_history.append(x2)
        target_history.append(target)
    
    results["kkt"] = {
        "t": t_history.copy(),
        "x1": x1_kkt_history.copy(),
        "x2": x2_kkt_history.copy(),
        "target": target_history.copy()
    }
    
    print(f"   Tiempo final: {sim_kkt.t:.3f}")
    print(f"   Error final x1: {abs(x1_kkt_history[-1]):.2e}")
    print(f"   Error final x2: {abs(x2_kkt_history[-1] - target_history[-1]):.2e}")
    
    # ===== MODO 2: BAUMGARTE PURO =====
    print("\n2. Ejecutando modo Baumgarte...")
    
    model_baum = create_mixed_constraint_system()
    model_baum.initialize(
        batch_size=B, device=device, dtype=torch.float32,
        initial_states={
            "mass1": {"x": x1_init.clone(), "v": v_init.clone()},
            "mass2": {"x": x2_init.clone(), "v": v_init.clone()}
        }
    )
    
    alg_baum = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30, verbose=False)
    solver_baum = TrapezoidalSolver(
        algebraic_solver=alg_baum,
        constraint_mode="kkt_baumgarte",
        baumgarte_alpha=2.0,
        baumgarte_beta=15.0
    )
    
    sim_baum = Simulator(model_baum, solver=solver_baum)
    
    # Simular
    x1_baum_history = []
    x2_baum_history = []
    
    for step in range(steps):
        sim_baum.step(dt, externals_fn=externals_fn)
        
        x1 = model_baum.states["mass1"][0, 0].item()
        x2 = model_baum.states["mass2"][0, 0].item()
        
        x1_baum_history.append(x1)
        x2_baum_history.append(x2)
    
    results["baumgarte"] = {
        "t": t_history.copy(),
        "x1": x1_baum_history,
        "x2": x2_baum_history,
        "target": target_history.copy()
    }
    
    print(f"   Tiempo final: {sim_baum.t:.3f}")
    print(f"   Error final x1: {abs(x1_baum_history[-1]):.2e}")
    print(f"   Error final x2: {abs(x2_baum_history[-1] - target_history[-1]):.2e}")
    
    # ===== MODO 3: MIXTO =====
    print("\n3. Ejecutando modo mixto (KKT + Baumgarte)...")
    
    model_mixed = create_mixed_constraint_system()
    model_mixed.initialize(
        batch_size=B, device=device, dtype=torch.float32,
        initial_states={
            "mass1": {"x": x1_init.clone(), "v": v_init.clone()},
            "mass2": {"x": x2_init.clone(), "v": v_init.clone()}
        }
    )
    
    alg_mixed = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30, verbose=False)
    solver_mixed = TrapezoidalSolver(
        algebraic_solver=alg_mixed,
        constraint_mode="kkt_mixed",
        lam_mask=[True, False],  # mass1_fixed=KKT, mass2_trajectory=Baumgarte
        baumgarte_alpha=2.0,
        baumgarte_beta=15.0
    )
    
    sim_mixed = Simulator(model_mixed, solver=solver_mixed)
    
    # Simular
    x1_mixed_history = []
    x2_mixed_history = []
    
    for step in range(steps):
        sim_mixed.step(dt, externals_fn=externals_fn)
        
        x1 = model_mixed.states["mass1"][0, 0].item()
        x2 = model_mixed.states["mass2"][0, 0].item()
        
        x1_mixed_history.append(x1)
        x2_mixed_history.append(x2)
    
    results["mixed"] = {
        "t": t_history.copy(),
        "x1": x1_mixed_history,
        "x2": x2_mixed_history,
        "target": target_history.copy()
    }
    
    print(f"   Tiempo final: {sim_mixed.t:.3f}")
    print(f"   Error final x1: {abs(x1_mixed_history[-1]):.2e}")
    print(f"   Error final x2: {abs(x2_mixed_history[-1] - target_history[-1]):.2e}")
    
    return results


def demonstrate_projection():
    """Demuestra proyección pos-paso con re-evaluación de externals."""
    
    print("\n=== DEMOSTRACIÓN: Proyección Pos-Paso ===\n")
    
    # Sistema simple: masa con restricción x = F_ext(t)
    model = Model("projection_demo")
    model.add_block("mass", Mass1D(m=1.0))
    model.connect("ForceExt", "mass.F")
    
    # Restricción: x debe seguir exactamente F_ext(t)
    def constraint_follow_force(t, states, inbuf, model, z):
        x = states["mass"][:, 0:1]
        F_ext = inbuf["mass"]["F"]
        return x - F_ext  # g = x - F_ext = 0
    
    model.add_constraint_eq("follow_external", constraint_follow_force)
    model.build()
    
    # Estados iniciales
    B = 1
    x0 = torch.tensor([[0.0]], dtype=torch.float32)
    v0 = torch.tensor([[0.0]], dtype=torch.float32)
    
    model.initialize(
        batch_size=B, device=torch.device("cpu"), dtype=torch.float32,
        initial_states={"mass": {"x": x0, "v": v0}}
    )
    
    # External que cambia con el tiempo: F_ext(t) = sin(2πt)
    def time_varying_force(t, k):
        return {"ForceExt": {"F": torch.tensor([[np.sin(2 * np.pi * t)]], dtype=torch.float32)}}
    
    # Simular SIN proyección
    from kombox.core.solvers import RK45Solver
    
    print("1. Simulación SIN proyección:")
    model_no_proj = Model("no_proj")
    model_no_proj.add_block("mass", Mass1D(m=1.0))
    model_no_proj.connect("ForceExt", "mass.F")
    model_no_proj.build()
    model_no_proj.initialize(
        batch_size=B, device=torch.device("cpu"), dtype=torch.float32,
        initial_states={"mass": {"x": x0.clone(), "v": v0.clone()}}
    )
    
    sim_no_proj = Simulator(model_no_proj, solver=RK45Solver())
    
    t_vals = []
    x_no_proj = []
    f_ext_vals = []
    
    for i in range(50):
        sim_no_proj.step(0.02, externals_fn=time_varying_force)
        t_vals.append(sim_no_proj.t)
        x_no_proj.append(model_no_proj.states["mass"][0, 0].item())
        f_ext_vals.append(np.sin(2 * np.pi * sim_no_proj.t))
    
    error_no_proj = np.abs(np.array(x_no_proj) - np.array(f_ext_vals))
    print(f"   Error promedio sin proyección: {np.mean(error_no_proj):.4f}")
    print(f"   Error máximo sin proyección: {np.max(error_no_proj):.4f}")
    
    # Simular CON proyección
    print("\n2. Simulación CON proyección:")
    model.states["mass"][:, 0:1] = x0.clone()  # Reset
    model.states["mass"][:, 1:2] = v0.clone()
    
    sim_proj = Simulator(model, solver=RK45Solver())
    sim_proj.enable_constraint_projection(
        enabled=True,
        tol=1e-8,
        max_iter=10,
        damping=1e-6,
        every_n_steps=1
    )
    
    x_with_proj = []
    
    for i in range(50):
        sim_proj.step(0.02, externals_fn=time_varying_force)
        x_with_proj.append(model.states["mass"][0, 0].item())
    
    error_with_proj = np.abs(np.array(x_with_proj) - np.array(f_ext_vals))
    print(f"   Error promedio con proyección: {np.mean(error_with_proj):.4f}")
    print(f"   Error máximo con proyección: {np.max(error_with_proj):.4f}")
    
    print(f"\n   Mejora en error promedio: {np.mean(error_no_proj) / np.mean(error_with_proj):.1f}x")


def test_jfnk_robustness():
    """Prueba robustez de JFNK con casos límite."""
    
    print("\n=== PRUEBA: Robustez JFNK ===\n")
    
    # Caso 1: Función independiente de z
    print("1. Testing VJP con función independiente de z...")
    
    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=5, verbose=False)
    
    def F_independent(z):
        # Función que no depende de z
        B = z.shape[0]
        return torch.ones((B, 3))  # Constante
    
    z_test = torch.randn(2, 4, requires_grad=True)
    v_test = torch.randn(2, 3)
    
    result = alg._vjp(F_independent, z_test, v_test)
    
    assert torch.allclose(result, torch.zeros_like(z_test))
    print("   ✓ VJP devuelve ceros correctamente para F independiente")
    
    # Caso 2: Solver con residual constante
    print("\n2. Testing solver con residual constante...")
    
    def F_constant_zero(z):
        return torch.zeros((z.shape[0], 2))
    
    z0 = torch.randn(3, 5)
    z_sol = alg.solve(F_constant_zero, z0)
    
    assert torch.isfinite(z_sol).all()
    print("   ✓ Solver maneja residual constante sin crashear")
    
    # Caso 3: Dimensión z vacía
    print("\n3. Testing con z de dimensión 0...")
    
    z_empty = torch.zeros(2, 0)
    z_result = alg.solve(lambda z: torch.zeros(z.shape[0], 1), z_empty)
    
    assert z_result.shape == z_empty.shape
    print("   ✓ Solver maneja z vacío correctamente")
    
    print("\n   Todas las pruebas de robustez pasaron ✓")


def plot_results(results):
    """Genera gráficos comparativos de los resultados."""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Comparación de Modos de Restricciones F3", fontsize=14)
        
        t = results["kkt"]["t"]
        target = results["kkt"]["target"]
        
        # Gráfico 1: Posición x1 (restricción fija)
        ax1.plot(t, results["kkt"]["x1"], 'b-', label="KKT puro", linewidth=2)
        ax1.plot(t, results["baumgarte"]["x1"], 'r--', label="Baumgarte", linewidth=2)
        ax1.plot(t, results["mixed"]["x1"], 'g:', label="Mixto", linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, label="Target")
        ax1.set_ylabel("Posición x₁")
        ax1.set_title("Masa 1: Restricción x₁ = 0")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Posición x2 (seguimiento de trayectoria)
        ax2.plot(t, results["kkt"]["x2"], 'b-', label="KKT puro", linewidth=2)
        ax2.plot(t, results["baumgarte"]["x2"], 'r--', label="Baumgarte", linewidth=2)
        ax2.plot(t, results["mixed"]["x2"], 'g:', label="Mixto", linewidth=2)
        ax2.plot(t, target, 'k-', alpha=0.7, label="Target: sin(t)", linewidth=1)
        ax2.set_ylabel("Posición x₂")
        ax2.set_title("Masa 2: Restricción x₂ = sin(t)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Error x1
        ax3.semilogy(t, np.abs(results["kkt"]["x1"]), 'b-', label="KKT puro")
        ax3.semilogy(t, np.abs(results["baumgarte"]["x1"]), 'r--', label="Baumgarte")
        ax3.semilogy(t, np.abs(results["mixed"]["x1"]), 'g:', label="Mixto")
        ax3.set_ylabel("Error |x₁|")
        ax3.set_xlabel("Tiempo")
        ax3.set_title("Error en Restricción x₁ = 0")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Error x2
        error_kkt = np.abs(np.array(results["kkt"]["x2"]) - np.array(target))
        error_baum = np.abs(np.array(results["baumgarte"]["x2"]) - np.array(target))
        error_mixed = np.abs(np.array(results["mixed"]["x2"]) - np.array(target))
        
        ax4.semilogy(t, error_kkt, 'b-', label="KKT puro")
        ax4.semilogy(t, error_baum, 'r--', label="Baumgarte")
        ax4.semilogy(t, error_mixed, 'g:', label="Mixto")
        ax4.set_ylabel("Error |x₂ - sin(t)|")
        ax4.set_xlabel("Tiempo")
        ax4.set_title("Error en Restricción x₂ = sin(t)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("f3_comparison_results.png", dpi=150, bbox_inches="tight")
        print("\n📊 Gráficos guardados en 'f3_comparison_results.png'")
        
    except ImportError:
        print("\n⚠️  Matplotlib no disponible - saltando generación de gráficos")


def main():
    """Función principal que ejecuta toda la demostración."""
    
    print("🚀 DEMOSTRACIÓN COMPLETA DEL HITO F3")
    print("=" * 50)
    print("Restricciones implícitas en Trapezoidal")
    print("- KKT puro, Baumgarte y modo mixto")
    print("- Re-evaluación de externals a t+dt")
    print("- Proyección pos-paso opcional")
    print("- Robustez JFNK con allow_unused")
    print("=" * 50)
    
    try:
        # Ejecutar comparación de modos
        results = run_comparison_study()
        
        # Demostrar proyección
        demonstrate_projection()
        
        # Probar robustez JFNK
        test_jfnk_robustness()
        
        # Generar gráficos
        plot_results(results)
        
        print("\n✅ DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print("\nCaracterísticas implementadas:")
        print("  • Tres modos de restricciones: KKT, Baumgarte, mixto")
        print("  • Re-evaluación automática de externals a t+dt")
        print("  • Proyección pos-paso con JVP/VJP + CG")
        print("  • JFNK robusto con allow_unused=True")
        print("  • Validaciones completas y mensajes de error claros")
        print("  • Compatibilidad con grafo de gradientes de PyTorch")
        
    except Exception as e:
        print(f"\n❌ Error durante la demostración: {e}")
        raise


if __name__ == "__main__":
    main()